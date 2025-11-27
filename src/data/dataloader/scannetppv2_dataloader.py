import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings
from sklearn.neighbors import KDTree
from torch_cluster import grid_cluster
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from collections.abc import Mapping, Sequence
import random
import pandas as pd

from data.collate.collate import point_collate_fn


class ScanNetPPV2DataSet(Dataset):
    """
    Dataset that loads the per gaussian labels derived from the point cloud labels provided in the Scannetppv2 dataset
    """

    def __init__(self, scene_names, scene_root, feature_root, metadata_root,label_root,device="cpu"):
        """
        scene_names: list of scene identifiers (str)
        scene_root: root folder where all files defining the 3D gaussian splats are stored
        feature_root: root folder where all files storing the per gaussian features of scenes are stored
        metadata_root: root folder where all files storing metadata of this dataset are stored

        """
        self.scene_names = scene_names
        self.scene_root = scene_root
        self.feature_root = feature_root
        self.metadata_root = metadata_root
        self.label_root = label_root

        self.text_embeddings_file = os.path.join(metadata_root,"text_embeddings.pth")
        self.device = device

        self.text_embeddings = torch.load(self.text_embeddings_file,weights_only=True).to(device)
        self.text_embeddings = F.normalize(self.text_embeddings, dim=-1)
        self.collate_fn = point_collate_fn

        self.num_classes = 100

        self.ignore_index = -1

    def __len__(self):
        return len(self.scene_names)

    def __getitem__(self, idx):
        scene_name = self.scene_names[idx]

        data = {}

        scene_dir = os.path.join(self.scene_root, scene_name)
        coord_file = os.path.join(scene_dir, "coord.npy")
        feat_file = os.path.join(self.feature_root, scene_name,"feat.pth")
        label_file = os.path.join(self.label_root,scene_name,"label.npy")

        data["coord"] = torch.Tensor(np.load(coord_file)).to(self.device)
        feat = torch.load(feat_file,weights_only=True).to(self.device)
        feat = F.normalize(feat, dim=-1)
        data["feat"] = feat @ self.text_embeddings.T  # logits

        N = data["coord"].shape[0]
        data["batch"] = torch.zeros(N, dtype=torch.long).to(self.device)
        data["grid_size"] = torch.Tensor([0.01]).to(self.device)

        labels = torch.Tensor(np.load(label_file)).to(self.device).long()
        valid_mask = torch.Tensor(labels != self.ignore_index).to(self.device)
        data["label"] = labels
        data["valid_mask"] = valid_mask
        data["ignore_index"] = self.ignore_index
        return data