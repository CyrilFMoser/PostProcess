import numpy as np
import os

root = "/cluster/home/cymoser/data/GaussianWorld/scannetpp_v2_mcmc_3dgs_preprocessed/train_grid1.0cm_chunk6x6_stride3x3"

counts = np.zeros(100)

for scene_name in os.listdir(root):
    segment_file = os.path.join(root,scene_name,"segment.npy")
    segment = np.load(segment_file)
    segment = segment[segment!=-1]
    counts += np.bincount(segment,minlength=100)

weights = counts.mean() / np.maximum(counts,1)
weights = weights.clip(0,50)
np.save(os.path.join("/cluster/home/cymoser/projects/PostProcess/data/scannetppv2/weights/weights.npy"),weights)