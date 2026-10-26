import math
import random

import numpy as np
import torch
from sklearn.metrics import f1_score, log_loss, roc_auc_score

def auc_acc(labels, pred):

    labels = np.asarray(labels).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    if not len(labels) or len(labels) != len(pred):
        raise ValueError()
    if not np.isfinite(pred).all():
        raise ValueError()
    auc = roc_auc_score(labels, pred) if len(np.unique(labels)) == 2 else float('nan')
    decisions = pred >= 0.5
    return float(auc), float(np.mean(decisions == labels)), float(f1_score(labels, decisions, zero_division=0))

def logLoss(labels, pred):
    return log_loss(labels, pred)

@torch.no_grad()
def score_pairs(model, users, items, batch_size=1024):
    if batch_size < 1:
        raise ValueError()
    scores = []
    for start in range(0, len(users), batch_size):
        user = torch.as_tensor(users[start:start + batch_size], dtype=torch.long)
        item = torch.as_tensor(items[start:start + batch_size], dtype=torch.long)
        prediction, _, _ = model(user, item)
        scores.append(prediction.detach().cpu().reshape(-1).numpy())
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)

@torch.no_grad()
def evaluate_split(model, interactions, batch_size=1024):

    model.eval()
    predictions = score_pairs(model, interactions[:, 0], interactions[:, 1], batch_size)
    labels = interactions[:, 2].detach().cpu().numpy()
    auc, acc, f1 = auc_acc(labels, predictions)
    return {'AUC': auc, 'ACC': acc, 'F1': f1, 'count': len(labels)}

def get_top_K_data(train_data, test_data, n_user, n_item):
    k_list = [1, 2, 3, 5, 10, 20]
    train_record = get_user_record(train_data.numpy(), True)
    test_record = get_user_record(test_data.numpy(), False)
    user_list = sorted(set(train_record) & set(test_record))
    item_set = set(range(n_user, n_user + n_item))
    return user_list, train_record, test_record, item_set, k_list

@torch.no_grad()
def topk_eval(model, train_data, test_data, n_user, n_item, batch_size=1024):
    user_list, train_record, test_record, item_set, k_list = get_top_K_data(
        train_data, test_data, n_user, n_item)
    model.eval()
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}
    hit_rate_list = {k: [] for k in k_list}
    for user in user_list:

        candidates = sorted(item_set - train_record[user] - test_record[user])
        negatives = random.Random(int(user)).sample(candidates, min(1000, len(candidates)))
        test_items = negatives + sorted(test_record[user])
        scores = score_pairs(model, np.full(len(test_items), user), np.asarray(test_items), batch_size)
        ranked = np.asarray(test_items)[np.argsort(-scores, kind='stable')].tolist()
        for k in k_list:
            top_items = ranked[:k]
            hits = len(set(top_items) & test_record[user])
            precision_list[k].append(hits / k)
            recall_list[k].append(hits / len(test_record[user]))
            dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(top_items)
                      if item in test_record[user])
            idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(test_record[user]))))
            ndcg_list[k].append(dcg / idcg if idcg else 0.0)
            hit_rate_list[k].append(float(hits > 0))
    def means(values):
        return [float(np.mean(values[k])) if values[k] else float('nan') for k in k_list]
    precision, recall = means(precision_list), means(recall_list)
    f1 = [2 * p * r / (p + r) if p + r > 0 else 0.0 for p, r in zip(precision, recall)]
    return precision, recall, f1, means(ndcg_list), means(hit_rate_list)

def get_user_record(data, is_train):
    record = {}
    for user, item, label in data:
        if is_train or label == 1:
            record.setdefault(int(user), set()).add(int(item))
    return record
