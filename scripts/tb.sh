#!/bin/bash

#SBATCH --account=a0104
#SBATCH --time=04:00:00

echo "Running on $(hostname -f)"
cd ~/projects/PostProcess
tensorboard --logdir runs
