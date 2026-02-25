#!/bin/bash
#SBATCH --uenv=pytorch/v2.6.0:v1
#SBATCH --view=default
#SBATCH --job-name=train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --account=a0104
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --exclusive

unset PYTHONPATH
export PYTHONUSERBASE="$(dirname "$(dirname "$(which python)")")"
export CUMM_CUDA_ARCH_LIST="9.0"
export SPCONV_DISABLE_JIT="1"
source /users/cymoser/projects/PostProcess/scripts/api.env
source /users/cymoser/projects/PostProcess/venv/bin/activate
export OMP_NUM_THREADS=4
ulimit -c 0

cd /users/cymoser/projects/PostProcess

# Pass all arguments through as Hydra overrides, e.g.:
#   sbatch run_train.sh model=full_skip class_subset=52 training.chkpt_newest=true
#   sbatch run_train.sh model=medium_skip training.lr=1e-4 run_tag=lowlr
python -m torch.distributed.run --nproc-per-node=4 src/utils/train.py "$@"
