#!/bin/bash
#SBATCH --uenv=pytorch/v2.6.0:v1 
#SBATCH --view=default 
#SBATCH --job-name=test_multi
#SBATCH --output=logs/test_multi.out
#SBATCH --error=logs/test_multi.err
#SBATCH --account=a0104
#SBATCH --time=1:00:00          # 12 hours
#SBATCH --gpus-per-task=4 
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

#unset PYTHONPATH
#export PYTHONUSERBASE="$(dirname "$(dirname "$(which python)")")"
#export CUMM_CUDA_ARCH_LIST="9.0"
#export SPCONV_DISABLE_JIT="1"
#source /users/cymoser/projects/PostProcess/venv/bin/activate
export OMP_NUM_THREADS=4
ulimit -c 0

export NCCL_NET="AWS Libfabric"
# Use GPU Direct RDMA when GPU and NIC are on the same NUMA node. More
# information about `NCCL_NET_GDR_LEVEL` can be found at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-net-gdr-level-formerly-nccl-ib-gdr-level
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
# Starting with nccl 2.27 a new protocol (LL128) was enabled by default, which
# typically performs worse on Slingshot. The following disables that protocol.
export NCCL_PROTO=^LL128
# These `FI` (libfabric) environment variables have been found to give the best
# performance on the Alps network across a wide range of applications. Specific
# applications may perform better with other values.
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd

#cd /users/cymoser/projects/PostProcess

#python -m torch.distributed.run --nproc-per-node=1 src/utils/train.py --class_subset=52 --preprocess



srun -ul bash -c '
    unset PYTHONPATH
    export PYTHONUSERBASE="$(dirname "$(dirname "$(which python)")")"
    export CUMM_CUDA_ARCH_LIST="9.0"
    export SPCONV_DISABLE_JIT="1"
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
    python -m torch.distributed.run ${TORCHRUN_ARGS} src/utils/train.py --class_subset=52 --chkpt_newest --nepoch=800 --model_typ=medium_skip_lowlr --lr=1e-5 --validate
'