import torch

class IoUMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.intersections = torch.zeros(self.num_classes, dtype=torch.float64)
        self.unions = torch.zeros(self.num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, preds, labels):
        # preds = [N] integer class predictions
        # labels = [N] integer ground truth labels
        
        for cls in range(self.num_classes):
            pred_mask = (preds == cls)
            label_mask = (labels == cls)

            intersection = (pred_mask & label_mask).sum().item()
            union = (pred_mask | label_mask).sum().item()

            self.intersections[cls] += intersection
            self.unions[cls] += union

    def compute(self):
        iou = self.intersections / (self.unions + 1e-8)
        # Ignore classes with no occurrences
        valid = self.unions > 0
        if valid.sum() == 0:
            return 0.0
        return iou[valid].mean().item()

    def combine(self,other):
        self.intersections += other.intersections
        self.unions += other.unions