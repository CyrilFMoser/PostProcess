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
from scipy.spatial import cKDTree

from data.collate.collate import point_collate_fn


class ScanNetPPV2DataSet(Dataset):
    """
    Dataset that loads the per gaussian labels derived from the point cloud labels provided in the Scannetppv2 dataset
    """

    def __init__(self,  scene_root, metadata_root,device="cpu",mode="train",subset=None,filtered_scenes=None,shuffle_classes=False,class_subset=100,sort_classes=True):
        """
        filtered_scenes: list of scene identifiers to filter out (str)
        scene_root: root folder where all files defining the 3D gaussian splats are stored
        feature_root: root folder where all files storing the per gaussian features of scenes are stored
        metadata_root: root folder where all files storing metadata of this dataset are stored

        """            
        if mode == "train":
            self.scene_root = os.path.join(scene_root,"train_grid1.0cm_chunk6x6_stride3x3")
        elif mode == "validation":
            self.scene_root = os.path.join(scene_root,"val")
        self.metadata_root = metadata_root
        self.mode = mode
        self.scene_names = os.listdir(self.scene_root)
        if filtered_scenes is not None:
            for scene in filtered_scenes:
                self.scene_names.remove(scene)

        self.text_embeddings_file = os.path.join(metadata_root,"scanetpp100_text_embeddings_siglip2.pt")
        self.shuffle_classes = shuffle_classes
        self.sort_classes = sort_classes
        self.device = device

        self.collate_fn = point_collate_fn

        self.num_classes = class_subset

        self.num_channels = self.num_classes + 11
        self.subset = subset
        self.ignore_index = -1

    def __len__(self):
        return len(self.scene_names)

    def __getitem__(self, idx):
        scene_name = self.scene_names[idx]
        #print(scene_name)
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
        coord = np.load(coord_file)
        data["coord"] = torch.Tensor(coord).to(self.device)
        pc_coord = np.load(pc_coord_file)
        data["pc_coord"] = torch.Tensor(pc_coord).to(self.device)
        subset_mask = None
        if self.subset is not None:
            subset_mask = self.subset_scene(coord)
        feat = torch.Tensor(np.load(feat_file)).to(self.device)
        if self.subset is not None:
            feat = feat[subset_mask]    
        norm = feat.norm(dim=-1, keepdim=True)
        feat.div_(norm.clamp_min_(1e-12))
        text_embeddings = torch.load(self.text_embeddings_file,weights_only=True).to(self.device)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        feat = feat @ text_embeddings.T  # cosine similarity
        del text_embeddings
        
        nc = feat.shape[1] # full number of classes
        base_pred = feat.argmax(dim=1)
        counts = torch.bincount(base_pred,minlength=nc)
        _, sorted_mask = torch.sort(counts,descending=True)
        feat = feat[:,sorted_mask]
        data["full_sorted_mask"]=sorted_mask
        # potentially subset number of classes here:
        feat = feat[:,:self.num_classes]
        if self.shuffle_classes:
            rng = np.random.default_rng()
            perm = rng.permutation(self.num_classes)
            feat[:] = feat[:,perm] 
            data["inv_perm"] = np.argsort(perm)
        else:
            data["inv_perm"] = np.arange(self.num_classes) # identity permutation

        if self.subset is not None:
            gs_features = [torch.Tensor(np.load(file)).to(self.device)[subset_mask] for file in [color_file,quat_file,scale_file]]
            gs_features.append(torch.Tensor(np.load(opacity_file)[subset_mask]).to(self.device).unsqueeze(1))
        else:
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
        data["valid_feat_mask"] = valid_feat_mask

        pc_labels = torch.Tensor(np.load(pc_segment_file)).to(self.device).long()[:,0]
        
        
        data["ignore_index"] = self.ignore_index
        
        # reorder the labels according to the rest
        mapping = torch.empty_like(sorted_mask).to(self.device)
        mapping[sorted_mask] = torch.arange(nc).to(self.device)
        gs_labels[valid_mask] = mapping[gs_labels[valid_mask]]
        data["segment"] = gs_labels
        data["valid_mask"] = valid_mask & (data["segment"] < self.num_classes)
        # reorder class order in pointclouds
        pc_valid_mask = pc_labels != self.ignore_index
        pc_labels[pc_valid_mask] = mapping[pc_labels[pc_valid_mask]]
        # ignore points that expect a class we don't have
        pc_labels[pc_labels >= self.num_classes] = self.ignore_index
        data["pc_segment"] = pc_labels

        if self.subset is not None:
            for key in ["segment","valid_mask","valid_feat_mask","batch","coord"]:
                data[key] = data[key][subset_mask]
            orig_tree = cKDTree(coord)
            orig_distances, _ = orig_tree.query(pc_coord)

            new_tree = cKDTree(data["coord"].cpu().numpy())
            new_distances, _  = new_tree.query(pc_coord)

            # only keep if their new nearest neighbour is not farther away 
            # then the max distance between two neighbours in the original setting
            mask = (new_distances <= orig_distances.max()) 
            for key in ["pc_segment","pc_coord"]:
                data[key] = data[key][mask]
        return data
    
    def subset_scene(self,coord):
        mask = np.ones(len(coord), dtype=bool)
        if len(mask) <=self.subset:
            return mask 
        mid_y = 0.5
        x_min = coord[:,0].min()
        x_max = coord[:,0].max()
        y_min = coord[:,1].min()
        y_max = coord[:,1].max()
        while mid_y >= 0.1:
            mid_x = 0.5
            while mid_x >= 0.1:
                for x_smaller in [0,1,2]:
                    if x_smaller==0: # x <= x_mid
                        x_mask = coord[:,0] <= (x_min + (x_max - x_min) * mid_x) 
                    elif x_smaller == 1: # x > x_mid
                        x_mask = coord[:,0] >  (x_max - (x_max - x_min) * mid_x)
                    else:
                        x_mask = np.ones(len(coord), dtype=bool)
                    for y_smaller in [0,1,2]:
                        if y_smaller==0: # y < y_mid
                            y_mask = coord[:,1] <= (y_min + (y_max - y_min) * mid_y) 
                        elif y_smaller==1: # y > y_mid
                            y_mask = coord[:,1] > (y_max - (y_max - y_min) * mid_y)
                        elif x_smaller==2: # can't ignore both
                            continue
                        else:
                            y_mask = np.ones(len(coord), dtype=bool)
                        cur_mask = x_mask & y_mask
                        if self.subset * 0.6 <= cur_mask.sum() < self.subset:
                            if self.mode=="train" and np.random.random(1)[0] <= 0.5:
                                return cur_mask
                            else:
                                mask = cur_mask
                                continue
                        if cur_mask.sum() < self.subset and cur_mask.sum() > mask.sum():
                            mask = cur_mask
                mid_x-=0.1
            mid_y -=0.1
        return mask