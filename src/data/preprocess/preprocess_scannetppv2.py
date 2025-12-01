import os
import open3d as o3d
import shutil

import numpy as np
import torch
from scipy.spatial import cKDTree
import time
import pandas as pd

# where the preprocessed gaussians splat scenes from scene_splat are stored
gs_root = "data/scenesplat/scannetppv2/scenes/train"
# where to store the per gaussian labels
label_root = "data/scannetppv2/gaussianlabels/train"
# where metadata is stored
metadata_root = "data/scannetppv2/metadata"

def main():


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    redo_everything = True

    if redo_everything:
        shutil.rmtree(label_root)
        os.mkdir(label_root)

    for scene_name in os.listdir(gs_root):
        if not redo_everything and scene_name in os.listdir(label_root):
            continue  # we already processed this
        embed_folder = os.path.join(label_root, scene_name)
        os.mkdir(embed_folder)

        # load the data for this scene
        gs_folder = os.path.join(gs_root, scene_name)
        # load the gaussian splat positions
        gs_coords_file = os.path.join(gs_folder, "coord.npy")
        gs_coords = np.load(gs_coords_file)
        # load the pointcloud positions and labels
        pc_segment_file = os.path.join(
            gs_folder, "pc_segment.npy")
        pc_coord_file = os.path.join(gs_folder,"pc_coord.npy")
        pc_coords = np.load(pc_coord_file)
        pc_labels = np.load(pc_segment_file)

        # remap the labels to top 100
        pc_labels = remap_labels(pc_labels[:,0])
        # find the nearest neighbours
        tree = cKDTree(pc_coords)
        _, indices = tree.query(gs_coords)

        gs_labels = pc_labels[indices]
        label_file = os.path.join(embed_folder, "label")
        np.save(label_file, gs_labels)

def remap_labels(labels):

    # -------------------------
    # Load data
    # -------------------------
    map_file = os.path.join(metadata_root,"map_benchmark.csv")
    classes_file = os.path.join(metadata_root,"semantic_classes.txt")
    top100_classes_file = os.path.join(metadata_root,"top100.txt")

    ignore_index = -1

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

    # -------------------------
    # Step 1: Build top100 lookup
    # -------------------------
    top100_lookup = {cls: idx for idx, cls in enumerate(top100_classes)}

    # -------------------------
    # Step 2: Map text_labels that are in top100_classes
    # -------------------------
    print(text_labels)
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

    N = len(labels)
    ignored_prev =(labels ==ignore_index).sum()
    ignored = (mapped_indices == ignore_index).sum()
    print(f"Previously ignored labels: {ignored_prev/N * 100.}%")
    print(f"Now ignored labels: {ignored/N * 100.}%")
    return mapped_indices

if __name__ == '__main__':
    main()
