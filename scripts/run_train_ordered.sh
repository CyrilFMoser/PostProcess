#!/bin/bash

#SBATCH --job-name=train_postprocess_skip_ordered
#SBATCH --output=train_postprocess_skip_ordered.out
#SBATCH --error=train_postprocess_skip_ordered.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH --gpus=rtx_4090:1            # request 1 GPU
#SBATCH --time=120:00:00          # adjust time as needed
#SBATCH --account=ls_polle

source ~/anaconda3/etc/profile.d/conda.sh

export PYTHONPATH=$(pwd):$PYTHONPATH

conda activate scene_splat
conda deactivate
conda deactivate
conda activate scene_splat

cd /cluster/home/cymoser/projects/PostProcess

python src/utils/train.py --class_subset=52 --load_chkpt --chkpt_path=/cluster/home/cymoser/cymoser/postprocess/ckpt/medium_skip52_classes_step_6118_epoch_22.pt