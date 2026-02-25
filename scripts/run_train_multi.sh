#!/bin/bash
#SBATCH --uenv=pytorch/v2.6.0:v1
#SBATCH --view=default
#SBATCH --job-name=train_multi
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=a0104
#SBATCH --time=12:00:00
#SBATCH --gpus-per-task=4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

export OMP_NUM_THREADS=4
ulimit -c 0

export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_PROTO=^LL128
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd

# Pass all arguments through as Hydra overrides, e.g.:
#   sbatch run_train_multi.sh model=full_skip class_subset=52 training.chkpt_newest=true
#   sbatch run_train_multi.sh model=medium_skip training.lr=1e-4 run_tag=lowlr
HYDRA_ARGS="$@"

srun -ul bash -c '
    unset PYTHONPATH
    export PYTHONUSERBASE="$(dirname "$(dirname "$(which python)")")"
    export CUMM_CUDA_ARCH_LIST="9.0"
    export SPCONV_DISABLE_JIT="1"
    source /users/cymoser/projects/PostProcess/scripts/api.env
    source /users/cymoser/projects/PostProcess/venv/bin/activate

    cd /users/cymoser/projects/PostProcess

    TORCHRUN_ARGS="\
    --master-addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --master-port=29500 \
    --node-rank=${SLURM_PROCID} \
    --nnodes=${SLURM_NNODES} \
    --nproc-per-node=${SLURM_GPUS_ON_NODE} \
    "
    echo $TORCHRUN_ARGS
    python -m torch.distributed.run ${TORCHRUN_ARGS} src/utils/train.py training.num_nodes=${SLURM_NNODES} '"$HYDRA_ARGS"'
'
