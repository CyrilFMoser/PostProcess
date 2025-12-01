import numpy as np
import torch
import os

# Truncate an already *preprocessed* scene to a smaller, more manageable size for testing

mode = "train"

scene_name = "09c1414f1b"
new_name = f"{scene_name}_mini"

scene_path = os.path.join("data/scenesplat/scannetppv2/scenes",mode)
scene_dir = os.path.join(scene_path, scene_name)
new_scene_dir = os.path.join(scene_path, new_name)

feat_root = os.path.join("data/scenesplat/scannetppv2/features",mode)
feat_dir = os.path.join(feat_root,scene_name)
new_feat_dir = os.path.join(feat_root,new_name)

label_root = os.path.join("data/scannetppv2/gaussianlabels",mode)
label_dir = os.path.join(label_root,scene_name)
new_label_dir = os.path.join(label_root,new_name)


N = 1000


if not os.path.exists(new_scene_dir):
    os.mkdir(new_scene_dir)


if not os.path.exists(new_feat_dir):
    os.mkdir(new_feat_dir)

if not os.path.exists(new_label_dir):
    os.mkdir(new_label_dir)

old_coord_file = os.path.join(scene_dir, "coord.npy")
new_coord_file = os.path.join(new_scene_dir, "coord.npy")

coord = np.load(old_coord_file)
np.save(new_coord_file, coord[:N])

old_label_file = os.path.join(label_dir,"label.npy")
new_label_file = os.path.join(new_label_dir,"label.npy")

label = np.load(old_label_file)
np.save(new_label_file,label[:N])

old_feat_file = os.path.join(feat_dir, f"feat.pth")
new_feat_file = os.path.join(new_feat_dir, f"feat.pth")
feat = torch.load(old_feat_file,weights_only=True)
torch.save(feat[:N], new_feat_file)
