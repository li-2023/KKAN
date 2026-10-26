import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils import data as Data

from data_loader import load_data
from deepfm import DeepFM
from evaluation import evaluate_split, topk_eval
from hkgr import HKGRO
from model import KnowledgeKan

def boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError()

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['KnowledgeKAN', 'DeepFM', 'HKGR-O'], default='KnowledgeKAN')
    parser.add_argument('--dataset', choices=['Amazon', 'MovieLens'], default='Amazon')
    parser.add_argument('--data_root', type=Path, default=None)
    parser.add_argument('--output_dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--KAN_model', type=str, default='KAN_RBF')
    parser.add_argument('--bases', nargs='+', default=None)
    parser.add_argument('--pre_model', type=int, choices=[1], default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dim', type=int, default=8)
    parser.add_argument('--n_sample', type=int, default=20)
    parser.add_argument('--n_sample_kg', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_ki_layers', type=int, default=2)
    parser.add_argument('--basis_size', type=int, default=5)
    parser.add_argument('--spline_order', type=int, default=3)
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--attention_temperature', type=float, default=1.0)
    parser.add_argument('--ieu', choices=['none', 'attention', 'gate'], default='none')
    parser.add_argument('--graph_batch_size', type=int, default=2048)
    parser.add_argument('--graph_checkpoint', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--deepfm_hidden', type=int, nargs='+', default=[128, 64, 32])
    parser.add_argument('--deepfm_dropout', type=float, default=0.5)
    parser.add_argument('--hkgr_dim', type=int, default=64)
    parser.add_argument('--hkgr_layers', type=int, default=1)
    parser.add_argument('--hin_layers', type=int, default=2)
    parser.add_argument('--component_dropout', type=float, default=0.1)
    parser.add_argument('--cl_weight', type=float, default=1e-4)
    parser.add_argument('--cl_temperature', type=float, default=0.1)
    parser.add_argument('--cl_chunk_size', type=int, default=512)
    parser.add_argument('--show_topk', type=boolean, nargs='?', const=True, default=True)
    parser.add_argument('--no-show_topk', dest='show_topk', action='store_false')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--seed', type=int, default=17)
    return parser

def build_model(args, n_user, n_entity, n_relation, interaction_graph, kg_graph):
    classes = {'KnowledgeKAN': KnowledgeKan, 'DeepFM': DeepFM, 'HKGR-O': HKGRO}
    return classes[args.model](args, n_user, n_entity, n_relation, interaction_graph, kg_graph)

def train_epoch(model, train_data, args, optimizer):
    model.train()
    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    totals = {'total': 0.0, 'recommendation': 0.0, 'contrastive': 0.0, 'regularization': 0.0}
    batches = Data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    for batch in batches:
        user = batch[:, 0].to(device=device, dtype=torch.long)
        item = batch[:, 1].to(device=device, dtype=torch.long)
        labels = batch[:, 2].to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, HKGRO):
            loss, terms = model.training_loss(user, item, labels)
        else:
            logits, _, _ = model(user, item, return_logits=True)
            loss = criterion(logits.reshape(-1), labels)
            terms = {'recommendation': loss.detach()}
        if not torch.isfinite(loss):
            raise FloatingPointError()
        loss.backward()
        optimizer.step()
        totals['total'] += loss.item()
        for name, value in terms.items():
            totals[name] += value.item()
    return totals

@torch.no_grad()
def evaluate(model, splits, n_user, n_item, args):
    train_data, eval_data, test_data = splits
    results = {}
    for name, interactions in (('train', train_data), ('eval', eval_data), ('test', test_data)):
        results[name] = evaluate_split(model, interactions, args.batch_size)
    if args.show_topk:
        metrics = topk_eval(model, train_data, test_data, n_user, n_item, args.batch_size)
        results['topk'] = dict(zip(('precision', 'recall', 'F1', 'NDCG', 'HR'), metrics))
        results['topk']['K'] = [1, 2, 3, 5, 10, 20]
    return results

@torch.no_grad()
def save_embeddings(model, args, root, run_name, n_user, n_item):
    model.eval()
    if isinstance(model, DeepFM):
        context, _ = model.kg_context()
        np.savez(root / 'emb' / f'{run_name}_embedding.npz',
                 user=model.user_embedding.weight.cpu().numpy(),
                 item=model.item_embedding.weight.cpu().numpy(),
                 entity=model.entity_embedding.weight.cpu().numpy(), kg_context=context.cpu().numpy())
    else:
        interaction, knowledge = model.graph_embeddings()
        interaction, knowledge = interaction.flatten(1), knowledge.flatten(1)
        entity = torch.cat((interaction[n_user:] + knowledge[:n_item], knowledge[n_item:]), dim=0)
        embeddings = torch.cat((interaction[:n_user], entity), dim=0).cpu().numpy()
        np.save(root / 'emb' / f'{run_name}_embedding.npy', embeddings)
        if args.model == 'KnowledgeKAN':

            np.save(root / 'emb' / f'{args.dataset}_embedding.npy', embeddings)

def main():
    parser = create_parser()
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.dim, args.graph_batch_size, *args.deepfm_hidden) < 1:
        parser.error('')
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.deepfm_dropout < 1:
        parser.error('')
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())
    if use_cuda and not torch.cuda.is_available():
        parser.error('')
    if use_cuda:
        torch.cuda.set_device(args.gpu_id)
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda', args.gpu_id) if use_cuda else torch.device('cpu')
    n_user, n_item, n_entity, n_relation, splits, cf_graph, kg_graph = load_data(args)
    model = build_model(args, n_user, n_entity, n_relation, cf_graph, kg_graph).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=0.0 if args.model == 'HKGR-O' else args.weight_decay)
    root = args.output_dir.resolve()
    for name in ('log', 'model', 'emb'):
        (root / name).mkdir(parents=True, exist_ok=True)
    if args.model == 'KnowledgeKAN':
        variant = '+'.join(args.bases) if args.bases else args.KAN_model
        run_name = f'{args.dataset}-{variant}-{args.dim}-{args.pre_model}'
    else:
        dimension = args.hkgr_dim if args.model == 'HKGR-O' else args.dim
        run_name = f'{args.dataset}-{args.model}-{dimension}-seed{args.seed}'
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(num_user=n_user, num_item=n_item, num_entity=n_entity, num_relation=n_relation)
    (root / 'model' / f'{run_name}.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'{args.model} {device} {n_user} {n_item}', flush=True)
    log_path = root / 'log' / f'{run_name}.txt'
    with log_path.open('a', encoding='utf-8') as log:
        log.write('\n\n' + str(args) + '\n')
        for epoch in range(args.epochs):
            losses = train_epoch(model, splits[0], args, optimizer)
            print(f'Epoch {epoch}: loss={losses["total"]:.6f}', flush=True)
            results = evaluate(model, splits, n_user, n_item, args)
            lines = [f'Epoch:{epoch}', f'loss: {losses}']
            for name in ('train', 'eval', 'test'):
                metrics = results[name]
                lines.append(f'{name}_AUC: {metrics["AUC"]:.4f}'
                             f'  {name}_ACC: {metrics["ACC"]:.4f}  {name}_F1: {metrics["F1"]:.4f}'
                             f'  samples: {metrics["count"]}')
            if args.show_topk:
                lines.append('topK:  ' + '  '.join(map(str, results['topk']['K'])))
                for name in ('precision', 'recall', 'F1', 'NDCG', 'HR'):
                    lines.append(name + ':  ' + '  '.join(f'{value:.6f}' for value in results['topk'][name]))
            message = '\n'.join(lines)
            print(message, flush=True)
            log.write(message + '\n')
            log.flush()
            with (root / 'log' / f'{run_name}.jsonl').open('a', encoding='utf-8') as history:
                history.write(json.dumps({'epoch': epoch, 'loss': losses, **results}) + '\n')
    save_embeddings(model, args, root, run_name, n_user, n_item)
    torch.save(model.state_dict(), root / 'model' / f'{run_name}.pt')
    print(root / 'model' / f'{run_name}.pt', flush=True)

if __name__ == '__main__':
    main()
