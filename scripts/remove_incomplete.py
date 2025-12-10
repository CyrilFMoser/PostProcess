import os
import shutil

path = "/cluster/home/cymoser/data/GaussianWorld/scannetpp_v2_mcmc_3dgs_preprocessed/train_grid1.0cm_chunk6x6_stride3x3"

scenes = os.listdir(path)
n_files = 13 
for scene in scenes:
    scene_dir = os.path.join(path,scene)
    if len(os.listdir(scene_dir)) < 13:
        shutil.rmtree(scene_dir)