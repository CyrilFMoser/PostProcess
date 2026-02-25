from scipy.spatial import KDTree
import torch
from scipy.stats import mode
import numpy as np
import time
from torch_cluster import knn

def neighbor_voting(neighbours, gs_pred, query, k=25):
    t0 = time.time()

    device = gs_pred.device
    gs_valid_pred = gs_pred
    
    num_query = query.shape[0]
    
    # gather neighbor predictions
   
    neighbor_pred = gs_valid_pred[neighbours.view(num_query*k)]  # shape: [num_query*k]
    
    # reshape to [num_query, k]
    neighbor_pred = neighbor_pred.view(num_query, k)
    
    valid_mask = neighbor_pred >= 0  # only real labels

    if valid_mask.any():
        max_label = neighbor_pred[valid_mask].max().item()
    else:
        # no valid neighbors at all
        print("No valid neighbours?!")
        return torch.full((num_query,), -1, device=device, dtype=torch.long)
    counts = torch.zeros((num_query, max_label+1), device=device, dtype=torch.int32)
    # vectorized accumulation
    t0_accum = time.time()
    rows = torch.arange(num_query, device=device)
    for i in range(k):
        col = neighbor_pred[:, i]
        mask = col >= 0
        counts[rows[mask], col[mask]] += 1

    top1_labels = torch.argmax(counts, dim=1)
    top1_labels[valid_mask.sum(dim=1) == 0] = -1
    t1_accum = time.time()
    t1 = time.time()
    #print(f"Total time: {t1-t0:.3f}s, Accum time: {t1_accum-t0_accum:.3f}s")

    return top1_labels

def neighbor_voting_uncached(gs_coords, gs_pred, query, k=25):
    t0 = time.time()

    device = gs_pred.device
    gs_valid_pred = gs_pred
    gs_valid_coords = gs_coords.to(device)

    t0_knn = time.time()

    # knn returns [2, num_query * k]
    edge_index = knn(gs_valid_coords, query.to(device), k)
    t1_knn = time.time()
    # edge_index[0] = indices in gs_valid_coords (neighbors)
    # edge_index[1] = indices in query (repeated)
    
    num_query = query.shape[0]
    
    # gather neighbor predictions
   
    neighbor_pred = gs_valid_pred[edge_index[1]]  # shape: [num_query*k]
    
    # reshape to [num_query, k]
    neighbor_pred = neighbor_pred.view(num_query, k)
    
    valid_mask = neighbor_pred >= 0  # only real labels

    if valid_mask.any():
        max_label = neighbor_pred[valid_mask].max().item()
    else:
        # no valid neighbors at all
        print("No valid neighbours?!")
        return torch.full((num_query,), -1, device=device, dtype=torch.long)
    counts = torch.zeros((num_query, max_label+1), device=device, dtype=torch.int32)
    # vectorized accumulation
    t0_accum = time.time()
    rows = torch.arange(num_query, device=device)
    for i in range(k):
        col = neighbor_pred[:, i]
        mask = col >= 0
        counts[rows[mask], col[mask]] += 1

    top1_labels = torch.argmax(counts, dim=1)
    top1_labels[valid_mask.sum(dim=1) == 0] = -1
    t1_accum = time.time()
    print(f"Total time: {t1_knn-t0:.3f}s, KNN time: {t1_knn-t0_knn:.3f}s, Accum time: {t1_accum-t0_accum:.3f}s")

    return top1_labels

def neighbor_voting_faster(gs_coords,gs_pred,query,valid_feat_mask=None):
    t0=time.time()
    gs_valid_pred = gs_pred
    gs_valid_coords = gs_coords.cpu()
    if valid_feat_mask is not None:
        mask = valid_feat_mask.cpu()
        gs_valid_pred = gs_valid_pred[mask]
        gs_valid_coords = gs_valid_coords[mask]
    t0_tree = time.time()
    tree = KDTree(gs_valid_coords)
    _,ind = tree.query(query.cpu(),k=25)
    t1_tree = time.time()
    ind_tensor = torch.from_numpy(ind).to(gs_pred.device)
    neighbor_pred = gs_valid_pred[ind_tensor].cpu().numpy()
    mask = neighbor_pred != -1
    max_label = neighbor_pred.max()
    counts = np.zeros((neighbor_pred.shape[0], max_label+1), dtype=int)
    
    t0_top1 = time.time()
    for i in range(neighbor_pred.shape[1]):   # loop over columns only
        col = neighbor_pred[:, i]
        col_mask = mask[:, i]
        rows = np.arange(neighbor_pred.shape[0])
        counts[rows[col_mask], col[col_mask]] += 1

    top1_labels = np.argmax(counts, axis=1)
    top1_labels[np.sum(mask, axis=1) == 0] = -1  # handle all -1 rows
    t1_top1 = time.time()

    t1 = time.time()
    print(f"\nTotal Time:{t1-t0},Tree Time:{t1_tree-t0_tree},Top1 Time:{t1_top1-t0_top1}\n")
    return torch.Tensor(top1_labels.flatten()).to(gs_pred.device)

def neighbor_voting_old(gs_coords,gs_pred,query,valid_feat_mask=None):
    t0=time.time()
    gs_valid_pred = gs_pred
    gs_valid_coords = gs_coords.cpu()
    if valid_feat_mask is not None:
        mask = valid_feat_mask.cpu()
        gs_valid_pred = gs_valid_pred[mask]
        gs_valid_coords = gs_valid_coords[mask]
    t0_tree = time.time()
    tree = KDTree(gs_valid_coords)
    _,ind = tree.query(query.cpu(),k=25)
    t1_tree = time.time()
    ind_tensor = torch.from_numpy(ind).to(gs_pred.device)
    neighbor_pred = gs_valid_pred[ind_tensor].cpu().numpy()
    top1_labels = []
    t0_top1 = time.time()
    for row in neighbor_pred:
        valid_preds = row[row != -1]
        if len(valid_preds) == 0:
            top1 = -1
        else:
            unique, counts = np.unique(valid_preds, return_counts=True)
            idx_sorted = np.argsort(-counts)
            sorted_labels = unique[idx_sorted]
            top1 = sorted_labels[0]
        top1_labels.append(top1)
    top1_labels = np.array(top1_labels)
    t1_top1 = time.time()
    t1 = time.time()
    print(f"\nTotal Time:{t1-t0},Tree Time:{t1_tree-t0_tree},Top1 Time:{t1_top1-t0_top1}\n")
    return torch.Tensor(top1_labels.flatten()).to(gs_pred.device)