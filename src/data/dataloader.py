import os
import torch
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import warnings


class SceneSplatDataset(Dataset):
    """
    Dataset for Processing the produced per Gaussian SceneSplat features
    """

    def __init__(self, scene_names, scene_root, feature_root, device="cpu"):
        """
        scene_names: list of scene identifiers (str)
        scene_root: root folder where all files defining the 3D gaussian splats are stored
        feature_root: root folder where all files storing the per gaussian features of scenes are stored
        """
        self.scene_names = scene_names
        self.scene_root = scene_root
        self.feature_root = feature_root
        self.device = device

        self.feature_dim = 768

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

        feat_file = os.path.join(feat_dir, f"{scene_name}_feat.pth")

        # Load data
        data = {}

        mask = np.load(valid_file).astype(bool)

        data["color"] = np.load(color_file)[mask]
        data["coord"] = np.load(coord_file)[mask]
        data["scale"] = np.load(scale_file)[mask]
        data["quat"] = np.load(quat_file)[mask]
        data["opacity"] = np.load(opacity_file)[mask]

        mask_t = torch.from_numpy(mask).to(self.device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            data["feature"] = torch.load(
                feat_file, map_location=self.device)[mask_t][:100]

        # Some postprocessing: convert to tensors if not already
        for key, value in data.items():
            if not isinstance(value, torch.Tensor):
                data[key] = torch.tensor(value, device=self.device)[:100]

        return data
