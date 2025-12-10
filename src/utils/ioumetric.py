import torch

class IoUMetric:
    def __init__(self, num_classes,excluded_classes=None):
        self.num_classes = num_classes
        self.excluded_classes =excluded_classes
        self.reset()

    def reset(self):
        self.intersections = torch.zeros(self.num_classes, dtype=torch.float64)
        self.unions = torch.zeros(self.num_classes, dtype=torch.float64)
        self.in_gts = torch.zeros(self.num_classes, dtype=torch.bool) # only if they appear in the GT
        self.gt = torch.zeros(self.num_classes,dtype=torch.float64)

    @torch.no_grad()
    def update(self, preds, labels):
        # preds = [N] integer class predictions
        # labels = [N] integer ground truth labels
        
        for cls in range(self.num_classes):
            if self.excluded_classes is not None and cls in self.excluded_classes:
                continue
            pred_mask = (preds == cls)
            label_mask = (labels == cls)
            in_gt =  label_mask.sum().item() != 0
            
            self.in_gts[cls] |= in_gt
            intersection = (pred_mask & label_mask).sum().item()
            union = (pred_mask | label_mask).sum().item()
            self.intersections[cls] += intersection
            self.unions[cls] += union
            self.gt[cls] += label_mask.sum().item()

    def compute(self):
        iou = self.intersections / (self.unions + 1e-8)
        # Ignore classes with no occurrences
        valid = (self.unions > 0) & self.in_gts
        if valid.sum() == 0:
            return 0.0
        return iou[valid].mean().item()

    def compute_acc(self):
        valid = self.in_gts
        if valid.sum() == 0:
            return 0.0
        acc = self.intersections[valid] / self.gt[valid]
        return acc.mean().item()

    def combine(self,other):
        self.intersections += other.intersections
        self.unions += other.unions
        self.in_gts |= other.in_gts
        self.gt += other.gt