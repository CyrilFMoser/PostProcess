import numpy as np
import open3d as o3d
import pandas as pd

# -------------------------
# Load data
# -------------------------
pcd_file = "data/scenesplat/scannetppv2/scenes/train/09c1414f1b/pc_segment.npy"
map_file = "data/scannetppv2/metadata/map_benchmark.csv"
classes_file = "data/scannetppv2/metadata/semantic_classes.txt"
top100_classes_file = "data/scannetppv2/metadata/top100.txt"

ignore_index = -1

#pcd = o3d.t.io.read_point_cloud(pcd_file)
#labels = pcd.point.label.numpy().flatten()

labels = np.load(pcd_file)[:,0].flatten()
print(labels)
# -------------------------
# Load map and semantic classes
# -------------------------
label_map = pd.read_csv(map_file)
label_map["class"] = label_map["class"].astype(str)

with open(classes_file, "r", encoding="utf-8") as f:
    classes = np.array([line.strip() for line in f], dtype=str)

with open(top100_classes_file, "r", encoding="utf-8") as f:
    top100_classes = np.array([line.strip() for line in f], dtype=str)

# -------------------------
# Build text label array
# -------------------------
text_labels = np.full_like(labels, fill_value="None", dtype=object)

valid_mask = labels != ignore_index
text_labels[valid_mask] = classes[labels[valid_mask]]

label_df = pd.DataFrame({"class": text_labels})

# -------------------------
# Step 1: Build top100 lookup
# -------------------------
top100_lookup = {cls: idx for idx, cls in enumerate(top100_classes)}

# -------------------------
# Step 2: Map text_labels that are in top100_classes
# -------------------------
text_labels_series = pd.Series(text_labels)  # convert to Series for vectorized ops

# Map labels that are in top100_classes
mapped_top100 = text_labels_series.map(top100_lookup)

# -------------------------
# Step 3: Map remaining labels using label_map
# -------------------------
# First, create a vectorized fallback map: map 'class' -> 'semantic_map_to' -> top100 index
# Only keep entries where semantic_map_to exists in top100_lookup
valid_map = label_map[label_map['semantic_map_to'].isin(top100_lookup)]
fallback_lookup = dict(zip(valid_map['class'], valid_map['semantic_map_to'].map(top100_lookup)))

# Apply fallback mapping to unmapped entries
final_mapping = mapped_top100.fillna(text_labels_series.map(fallback_lookup))

# -------------------------
# Step 4: Convert to numpy array, with ignore_index for unmapped
# -------------------------
mapped_indices = final_mapping.to_numpy()
mapped_indices = np.where(pd.isna(mapped_indices), ignore_index, mapped_indices).astype(int)

print(f"final:{mapped_indices}")
print(f"Initial ignored labels: {(labels ==ignore_index).sum()}")
print(f"New ignored labels: {(mapped_indices == ignore_index).sum()}")
