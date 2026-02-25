from scipy.spatial import KDTree
import torch
from scipy.stats import mode
import numpy as np

def neighbor_voting(data:dict,gs_pred,query,valid_feat_mask=None):
    gs_coords = data["coord"].cpu()
    gs_valid_pred = gs_pred
    gs_valid_coords = gs_coords
    if valid_feat_mask is not None:
        mask = valid_feat_mask.cpu()
        gs_valid_pred = gs_valid_pred[mask]
        gs_valid_coords = gs_valid_coords[mask]
        
    tree = KDTree(gs_valid_coords)
    _,ind = tree.query(query.cpu(),k=25)
    ind_tensor = torch.from_numpy(ind).to(gs_pred.device)
    neighbor_pred = gs_valid_pred[ind_tensor].cpu().numpy()
    top1_labels = []
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
    
    return torch.Tensor(top1_labels.flatten()).to(gs_pred.device)