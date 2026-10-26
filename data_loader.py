from pathlib import Path

import numpy as np
import torch

def sample_neighbors(graph_dict, node_ids, neighbor_num, with_relations=False):
    neighbors = np.full((len(node_ids), neighbor_num), -1, dtype=np.int64)
    relations = np.full_like(neighbors, -1) if with_relations else None
    for row, node_id in enumerate(node_ids):
        values = graph_dict.get(node_id, [])
        if not values:
            continue
        indices = np.random.choice(len(values), neighbor_num, replace=len(values) < neighbor_num)
        sampled = np.asarray(values, dtype=np.int64)[indices]
        if with_relations:
            neighbors[row] = sampled[:, 0]
            relations[row] = sampled[:, 1]
        else:
            neighbors[row] = sampled
    if with_relations:
        return torch.from_numpy(neighbors), torch.from_numpy(relations)
    return torch.from_numpy(neighbors)

def load_data(args):
    data_root = getattr(args, 'data_root', None)
    data_root = Path(data_root) if data_root else Path(__file__).resolve().parent.parent / 'data'
    file_path = data_root / args.dataset
    train_data = torch.from_numpy(np.loadtxt(file_path / 'train.txt', dtype=np.int32, ndmin=2))
    test_data = torch.from_numpy(np.loadtxt(file_path / 'test.txt', dtype=np.int32, ndmin=2))
    eval_data = torch.from_numpy(np.loadtxt(file_path / 'eval.txt', dtype=np.int32, ndmin=2))
    rating_np = np.concatenate((train_data.numpy(), test_data.numpy(), eval_data.numpy()), axis=0)
    num_user = len(np.unique(rating_np[:, 0]))
    item_set = set(rating_np[:, 1].tolist())
    num_item = len(item_set)
    if set(rating_np[:, 0]) != set(range(num_user)):
        raise ValueError()
    if item_set != set(range(num_user, num_user + num_item)):
        raise ValueError()
    if not np.isin(rating_np[:, 2], [0, 1]).all():
        raise ValueError()
    if getattr(args, 'model', 'KnowledgeKAN') != 'KnowledgeKAN':
        interaction_graph = normalized_interaction_graph(train_data, num_user, num_item)
        kg_graph, num_entity, num_relation = full_kg_graph(file_path / 'kg.txt', num_user, item_set)
        return num_user, num_item, num_entity, num_relation, (train_data, eval_data, test_data), interaction_graph, kg_graph
    interaction_graph, _ = load_interaction_graph(file_path / 'train.txt', args.n_sample, num_user, num_item)
    kg_sample = getattr(args, 'n_sample_kg', None) or args.n_sample
    kg_graph, _, num_entity, num_relation = load_kg_graph(file_path / 'kg.txt', kg_sample, num_user, item_set)
    return num_user, num_item, num_entity, num_relation, (train_data, eval_data, test_data), interaction_graph, kg_graph

def load_kg_graph(file, neighbor_num, num_user, item_set):
    graph_np = np.unique(np.loadtxt(file, dtype=np.int64, ndmin=2), axis=0)
    entity_set = set(graph_np[:, 0]) | set(graph_np[:, 1]) | item_set
    num_entity = int(max(entity_set)) - num_user + 1
    num_relation = int(graph_np[:, 2].max()) + 1
    graph_dict = {}
    for source, target, relation in graph_np:
        graph_dict.setdefault(int(source), []).append((int(target), int(relation)))
        if source != target:
            graph_dict.setdefault(int(target), []).append((int(source), int(relation)))
    neighbors, relations = sample_neighbors(graph_dict, range(num_user, num_user + num_entity), neighbor_num, True)
    local_items = neighbors.numpy() - num_user
    item_mask = (local_items >= 0) & (local_items < len(item_set))
    item_mask[:len(item_set)] = False
    rows, columns = np.nonzero(item_mask)
    context_indices = torch.from_numpy(np.stack((rows, local_items[rows, columns])))
    degree = item_mask.sum(axis=1)
    context_values = torch.from_numpy((1.0 / degree[rows]).astype(np.float32))
    context = torch.sparse_coo_tensor(context_indices, context_values, (num_entity, len(item_set))).coalesce()
    return (neighbors, relations, context), graph_dict, num_entity, num_relation

def load_interaction_graph(file, neighbor_num, num_user, num_item):
    graph_np = np.loadtxt(file, dtype=np.int64, ndmin=2)
    positive_pairs = np.unique(graph_np[graph_np[:, 2] > 0, :2], axis=0)
    graph_dict = {}
    for source, target in positive_pairs:
        graph_dict.setdefault(int(source), []).append(int(target))
        graph_dict.setdefault(int(target), []).append(int(source))
    neighbors = sample_neighbors(graph_dict, range(num_user + num_item), neighbor_num)
    return neighbors, graph_dict

def normalized_interaction_graph(train_data, num_user, num_item):

    pairs = np.unique(train_data.numpy()[train_data[:, 2].numpy() > 0, :2], axis=0)
    edges = np.concatenate((pairs, pairs[:, ::-1]), axis=0).T.copy()
    degree = np.bincount(edges[0], minlength=num_user + num_item)
    values = (1.0 / np.sqrt(degree[edges[0]] * degree[edges[1]])).astype(np.float32)
    return torch.sparse_coo_tensor(torch.from_numpy(edges), torch.from_numpy(values),
                                  (num_user + num_item, num_user + num_item)).coalesce()

def full_kg_graph(file, num_user, item_set):

    triples = np.loadtxt(file, dtype=np.int64, ndmin=2)
    if not len(triples):
        return (torch.empty(2, 0, dtype=torch.long), torch.empty(0, dtype=torch.long)), len(item_set), 0
    if triples[:, :2].min() < num_user or triples[:, 2].min() < 0:
        raise ValueError()
    triples[:, :2] -= num_user
    reverse = triples[:, [1, 0, 2]]
    triples = np.unique(np.concatenate((triples, reverse), axis=0), axis=0)
    num_entity = max(int(triples[:, :2].max()) + 1, len(item_set))
    num_relation = int(triples[:, 2].max()) + 1
    return (torch.from_numpy(triples[:, :2].T.copy()), torch.from_numpy(triples[:, 2].copy())), num_entity, num_relation
