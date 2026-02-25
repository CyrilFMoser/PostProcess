import os
import shutil

val = True
path = "/users/cymoser/scratch/GaussianWorld/train_grid1.0cm_chunk6x6_stride3x3"
n_files = 13 
if val:
    path = "/users/cymoser/scratch/GaussianWorld/val"
    n_files = 15
scenes = os.listdir(path)
count = 0
for scene in scenes:
    scene_dir = os.path.join(path,scene)
    if len(os.listdir(scene_dir)) < n_files:
        shutil.rmtree(scene_dir)
        count+=1
print(f"Deleted {count} incomplete scene folders")
