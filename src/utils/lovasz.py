"""
Lovász-Softmax loss for multi-class segmentation.
Adapted from https://arxiv.org/abs/1705.08790
(copied locally to avoid the pointcept package import chain)
"""

from itertools import filterfalse
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_softmax_flat(probas, labels, classes="present", class_seen=None):
    if probas.numel() == 0:
        return probas.sum()  # scalar 0.0, preserves grad graph
    C = probas.size(1)
    losses = []
    for c in labels.unique():
        if class_seen is None or c in class_seen:
            fg = (labels == c).type_as(probas)
            if classes == "present" and fg.sum() == 0:
                continue
            class_pred = probas[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            fg_sorted = fg[perm.data]
            losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return _mean(losses)


def _flatten_probas(probas, labels, ignore=None):
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    C = probas.size(1)
    probas = torch.movedim(probas, 1, -1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    return probas[valid], labels[valid]


def _mean(values, empty=0):
    values = list(values)
    if not values:
        return empty
    return sum(values) / len(values)


class LovaszLoss(_Loss):
    def __init__(
        self,
        mode: str = "multiclass",
        per_image: bool = False,
        ignore_index: Optional[int] = None,
        loss_weight: float = 1.0,
    ):
        assert mode == "multiclass", "Only multiclass mode is supported here"
        super().__init__()
        self.per_image = per_image
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def forward(self, y_pred, y_true):
        probas = y_pred.softmax(dim=1)
        probas_flat, labels_flat = _flatten_probas(probas, y_true, self.ignore_index)
        loss = _lovasz_softmax_flat(probas_flat, labels_flat)
        return loss * self.loss_weight
