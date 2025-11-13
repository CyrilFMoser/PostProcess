import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings
from sklearn.neighbors import KDTree
from torch_cluster import grid_cluster
from torch_scatter import scatter_mean
import torch.nn.functional as F

from data.tree import getTree


class SceneSplatDataset(Dataset):
    """
    Dataset for Processing the produced per Gaussian SceneSplat features
    """

    def __init__(self, scene_names, scene_root, feature_root, text_embeddings_root, device="cpu", ignore_index=-1):
        """
        scene_names: list of scene identifiers (str)
        scene_root: root folder where all files defining the 3D gaussian splats are stored
        feature_root: root folder where all files storing the per gaussian features of scenes are stored
        """
        self.scene_names = scene_names
        self.scene_root = scene_root
        self.feature_root = feature_root
        self.text_embeddings_root = text_embeddings_root
        self.device = device

        self.ignore_index = ignore_index

        self.percentage = 0.1

        self.feature_dim = 768
        self.voxelW = 0.3
        self.voxel_size = torch.Tensor(
            [self.voxelW, self.voxelW, self.voxelW]).to(device)

        text_embeddings_file = os.path.join(
            text_embeddings_root, "top100_text_embeddings_siglip2.pt")

        self.text_embeddings = torch.load(
            text_embeddings_file, weights_only=True).to(device)
        self.text_embeddings = F.normalize(self.text_embeddings)

    def __len__(self):
        return len(self.scene_names)

    def __getitem__(self, idx):
        scene_name = self.scene_names[idx]
        scene_dir = os.path.join(self.scene_root, scene_name)
        feat_dir = self.feature_root

        # Locate files
        color_file = os.path.join(scene_dir, "color.npy")
        coord_file = os.path.join(scene_dir, "coord.npy")
        scale_file = os.path.join(scene_dir, "scale.npy")
        quat_file = os.path.join(scene_dir, "quat.npy")
        opacity_file = os.path.join(scene_dir, "opacity.npy")
        valid_file = os.path.join(scene_dir, "valid_feat_mask.npy")
        segment_file = os.path.join(scene_dir, "segment.npy")

        feat_file = os.path.join(feat_dir, f"{scene_name}_feat.pth")

        # Load data
        data = {}

        data["color"] = np.load(color_file)
        data["coord"] = np.load(coord_file)
        data["scale"] = np.load(scale_file)
        data["quat"] = np.load(quat_file)
        data["opacity"] = np.load(opacity_file)
        data["segment_raw"] = np.load(segment_file).astype(np.int64)
        data["valid_mask"] = np.load(valid_file).astype(bool)

        max_index = int(len(data["color"])*self.percentage)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            data["feature"] = torch.load(
                feat_file, map_location=self.device)

        # Some postprocessing: convert to tensors if not already
        for key, value in data.items():
            if not isinstance(value, torch.Tensor):
                data[key] = torch.tensor(value, device=self.device)
            if self.percentage < 1:
                data[key] = data[key][:max_index]

        coord = data["coord"]

        cluster = grid_cluster(coord, self.voxel_size)
        unique_clusters, cluster_idx = torch.unique(
            cluster, return_inverse=True)

        voxel_feature = scatter_mean(data["feature"], cluster_idx, dim=0)

        min_coord = coord.min(dim=0).values  # min bounding box corner
        voxel_coord_int = torch.floor(
            (coord-min_coord)/self.voxelW).int()
        unique_voxels, inverse_voxel_idx = torch.unique(
            voxel_coord_int, dim=0, return_inverse=True)

        # add voxel center and go back to original coordinate space
        voxel_coord = (unique_voxels.float() + 0.5) * self.voxelW + min_coord

        # assign textembeddings to the fitting gaussians

        segment_raw = data["segment_raw"][:, 0]
        valid_segment = segment_raw[data["valid_mask"]]
        segment = valid_segment[valid_segment != self.ignore_index]

        # [V, F_in] where V is number of valid, non ignore_index gaussians
        gaussian_text_embedding = self.text_embeddings[segment]

        data["coord"] = data["coord"].permute(1, 0)
        data["feature"] = data["feature"].permute(1, 0)
        data["voxel_feature"] = voxel_feature.permute(1, 0)
        data["voxel_coord"] = voxel_coord.permute(1, 0)
        data["voxel_to_gauss"] = cluster_idx
        data["gaussian_text_embedding"] = gaussian_text_embedding.permute(
            1, 0)  # [F_in, V]

        return data
