import os

def write_scene_list(folder, outfile, skip_first=0):
    # List all directory names inside the folder
    scenes = sorted([
        name for name in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, name))
    ])

    # Skip first X
    scenes = scenes[skip_first:]

    # Write to text file
    with open(outfile, "w") as f:
        for s in scenes:
            f.write(s + "\n")

    print(f"Wrote {len(scenes)} scenes to {outfile}")


# ---- CONFIG ----
training = "/cluster/home/cymoser/data/GaussianWorld/scannetpp_v2_mcmc_3dgs_preprocessed/train_grid1.0cm_chunk6x6_stride3x3"
validation = "/cluster/home/cymoser/data/GaussianWorld/scannetpp_v2_mcmc_3dgs_preprocessed/val"
N = 10  # how many scenes we want in total

write_scene_list(training, "/cluster/home/cymoser/projects/PostProcess/data/scannetppv2/filtered_scenes/train.txt", skip_first=int(N*0.8))
write_scene_list(validation, "/cluster/home/cymoser/projects/PostProcess/data/scannetppv2/filtered_scenes/validation.txt", skip_first=int(N*0.2))
