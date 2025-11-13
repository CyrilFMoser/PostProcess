import numpy as np
import open3d as o3d

file = "/mnt/g/Projects/RICS/PostProcess/data/pointclouds/13c3e046d7/mesh_aligned_0.05_semantic.ply"

pcd = o3d.t.io.read_point_cloud(file)
print(pcd.point)  # lists all available attributes
