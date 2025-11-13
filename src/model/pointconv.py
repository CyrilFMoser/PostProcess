"""
Classification Model
Author: Wenxuan Wu
Date: September 2019
"""
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from utils.pointconv_util import PointConvDensitySetAbstraction


class PointConvDensityClsSsg(nn.Module):
    def __init__(self, feature_dim, num_classes=40):
        super(PointConvDensityClsSsg, self).__init__()
        self.F_in = feature_dim
        self.F1 = 512
        self.F2 = 256
        self.F3 = 128
        self.F4 = 128
        self.F5 = 64

        self.sa1 = PointConvDensitySetAbstraction(
            npoint=None, nsample=32, in_channel=self.F_in+3,  mlp=[self.F1, self.F1, self.F2], bandwidth=0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(
            npoint=None, nsample=32, in_channel=self.F2+3, mlp=[self.F2, self.F2, self.F3], bandwidth=0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(
            npoint=None, nsample=32, in_channel=self.F3+3, mlp=[self.F3, self.F4, self.F5], bandwidth=0.4, group_all=False)

        self.final_linear = nn.Linear(self.F5, self.F_in)

        self.relu = nn.ReLU()

    def forward(self, data):
        xyz = data["coord"]
        feat = data["feature"]
        B, _, N = xyz.shape  # N = number of original gaussians

        voxel_xyz = data["voxel_coord"]
        voxel_feat = data["voxel_feature"]
        _, _, M = voxel_xyz.shape  # M = number of voxels
        print(f"M:{M}")
        l1_xyz, l1_points = self.sa1(voxel_xyz, voxel_feat)
        print(voxel_xyz.shape)
        print(l1_xyz.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, M, self.F5)
        new_voxel_feat = self.relu(self.final_linear(x))

        # add the voxel feature to all the gaussian features contained in that voxel
        voxel_to_gauss = data["voxel_to_gauss"]
        if len(voxel_to_gauss) == 0:
            return new_voxel_feat

        # [B, M, F_in]
        voxel_to_gauss_exp = voxel_to_gauss.unsqueeze(
            -1).expand(-1, -1, self.F_in)

        gauss_voxel_feat = torch.gather(
            new_voxel_feat, dim=1, index=voxel_to_gauss_exp)  # [B, N, F_in]
        gauss_voxel_feat = gauss_voxel_feat.permute(0, 2, 1)  # [B, F_in, N]

        # Add to original Gaussian features
        new_feat = feat + gauss_voxel_feat

        return new_feat


if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8, 3, 2048))
    label = torch.randn(8, 16)
    model = PointConvDensityClsSsg(num_classes=40)
    output = model(input)
    print(output.size())
