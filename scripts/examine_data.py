import numpy as np
import open3d as o3d

file = "/mnt/g/Projects/RICS/PostProcess/data/scannetppv2/pointclouds/13c3e046d7/mesh_aligned_0.05_semantic.ply"
# file = "/mnt/g/Projects/RICS/PostProcess/data/scenesplat/scannetppv2/scenes/13c3e046d7/pc_coord.npy"

if ".ply" in file:
    pcd = o3d.t.io.read_point_cloud(file)
    print(pcd.point)  # lists all available attributes
elif ".npy" in file:
    data = np.load(file)
    print(data.shape)
    print(data)
