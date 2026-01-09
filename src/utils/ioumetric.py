import torch

class IoUMetric:
    def __init__(self, num_classes,subset_num_classes=None,excluded_classes=None):
        self.num_classes = num_classes
        self.excluded_classes =excluded_classes
        self.subset_num_classes = subset_num_classes
        if self.subset_num_classes is None:
            self.subset_num_classes = self.num_classes
        self.reset()

    def reset(self):
        self.intersections = torch.zeros(self.num_classes, dtype=torch.float64)
        self.unions = torch.zeros(self.num_classes, dtype=torch.float64)
        self.in_gts = torch.zeros(self.num_classes, dtype=torch.bool) # only if they appear in the GT
        self.gt = torch.zeros(self.num_classes,dtype=torch.float64)

    @torch.no_grad()
    def update(self, preds, labels,full_sorted_mask):
        # preds = [N] integer class predictions
        # labels = [N] integer ground truth labels
        # full_sorted_mask = [C] integer remapping used
        
        for orig_cls in range(self.num_classes):
            cls = full_sorted_mask[orig_cls]
            if cls >= self.subset_num_classes:
                continue
            pred_mask = (preds == cls)
            label_mask = (labels == cls)
            in_gt =  label_mask.sum().item() != 0
            if not in_gt:
                continue
            self.in_gts[orig_cls] = in_gt
            intersection = (pred_mask & label_mask).sum().item()
            union = (pred_mask | label_mask).sum().item()
            self.intersections[orig_cls] += intersection
            self.unions[orig_cls] += union
            self.gt[orig_cls] += label_mask.sum().item()
    def compute(self):
        iou = self.intersections / (self.unions + 1e-8)
        # Ignore classes with no occurrences
        valid = (self.unions > 0) & self.in_gts
        valid[self.excluded_classes] = False
        if valid.sum() == 0:
            return 0.0
        return iou[valid].mean().item()

    def compute_acc(self):
        valid = self.in_gts.detach().clone()
        valid[self.excluded_classes] = False
        if valid.sum() == 0:
            return 0.0
        acc = self.intersections[valid] / self.gt[valid]
        return acc.mean().item()

    def combine(self,other):
        self.intersections += other.intersections
        self.unions += other.unions
        self.in_gts = self.in_gts | other.in_gts
        self.gt += other.gt

    def log_all(self,prefix,writer,label_map,global_step):
        valid = self.in_gts
        if valid.sum() == 0:
            return 0.0
        iou = self.intersections / (self.unions + 1e-8)
        for cls in range(self.num_classes):
            if not valid[cls]:
                continue
            name = label_map[cls]
            writer.add_scalar(prefix+name,iou[cls],global_step)
