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
import pandas as pd

from data.collate.collate import point_collate_fn


class ScanNetPPV2DataSet(Dataset):
    """
    Dataset that loads the per gaussian labels derived from the point cloud labels provided in the Scannetppv2 dataset
    """

    def __init__(self, scene_names, scene_root, metadata_root,device="cpu",mode="train",subset=None):
        """
        scene_names: list of scene identifiers (str)
        scene_root: root folder where all files defining the 3D gaussian splats are stored
        feature_root: root folder where all files storing the per gaussian features of scenes are stored
        metadata_root: root folder where all files storing metadata of this dataset are stored

        """            
        self.scene_names = scene_names
        if mode == "train":
            self.scene_root = os.path.join(scene_root,"train_grid1.0cm_chunk6x6_stride3x3")
        elif mode == "validation":
            self.scene_root = os.path.join(scene_root,"val")
        self.metadata_root = metadata_root

        if self.scene_names is None:
            self.scene_names = os.listdir(self.scene_root)

        self.text_embeddings_file = os.path.join(metadata_root,"scanetpp100_text_embeddings_siglip2.pt")
        self.device = device

        self.collate_fn = point_collate_fn

        self.num_classes = 100

        self.num_channels = self.num_classes + 11
        self.subset = subset
        self.ignore_index = -1

    def __len__(self):
        return len(self.scene_names)

    def __getitem__(self, idx):
        scene_name = self.scene_names[idx]

        data = {}
        data["name"] = scene_name

        scene_dir = os.path.join(self.scene_root, scene_name)
        coord_file = os.path.join(scene_dir, "coord.npy")
        color_file = os.path.join(scene_dir,"color.npy") # 3
        quat_file = os.path.join(scene_dir,"quat.npy") # 4
        scale_file = os.path.join(scene_dir,"scale.npy") # 3
        opacity_file = os.path.join(scene_dir,"opacity.npy") # 1
        feat_file = os.path.join(scene_dir,"lang_feat.npy")
        segment_file = os.path.join(scene_dir,"segment.npy")
        pc_coord_file = os.path.join(scene_dir,"pc_coord.npy")
        pc_segment_file = os.path.join(scene_dir,"pc_segment.npy")
        valid_feat_mask_file = os.path.join(scene_dir,"valid_feat_mask.npy")
        data["coord"] = torch.Tensor(np.load(coord_file)).to(self.device)
        data["pc_coord"] = torch.Tensor(np.load(pc_coord_file)).to(self.device)

        feat = torch.Tensor(np.load(feat_file)).to(self.device)
        feat = F.normalize(feat, dim=-1)
        text_embeddings = torch.load(self.text_embeddings_file,weights_only=True).to(self.device)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        feat = feat @ text_embeddings.T  # cosine similarity
        del text_embeddings

        gs_features = [torch.Tensor(np.load(file)).to(self.device) for file in [color_file,quat_file,scale_file]]
        gs_features.append(torch.Tensor(np.load(opacity_file)).to(self.device).unsqueeze(1))
        feat_list = [feat]
        feat_list.extend(gs_features)
        feat = torch.cat(feat_list,dim=1)
        data["feat"] = feat
        del feat_list

        N = data["coord"].shape[0]
        data["batch"] = torch.zeros(N, dtype=torch.long).to(self.device)
        data["grid_size"] = torch.Tensor([0.01]).to(self.device)
        gs_labels = torch.Tensor(np.load(segment_file)).to(self.device).long()[:,0]
        valid_mask = torch.Tensor(gs_labels != self.ignore_index).to(self.device)
        valid_feat_mask = torch.Tensor(np.load(valid_feat_mask_file)).to(self.device).to(bool)
        data["segment"] = gs_labels
        data["valid_mask"] = valid_mask
        data["valid_feat_mask"] = valid_feat_mask

        pc_labels = torch.Tensor(np.load(pc_segment_file)).to(self.device).long()[:,0]
        
        data["pc_segment"] = pc_labels
        
        data["ignore_index"] = self.ignore_index
        if self.subset is not None:
            for key in ["segment","valid_mask","valid_feat_mask","batch","feat","coord"]:
                data[key] = data[key][:self.subset]
        return data