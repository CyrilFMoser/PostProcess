uenv start --view=default pytorch/v2.6.0:v1
unset PYTHONPATH
export PYTHONUSERBASE="$(dirname "$(dirname "$(which python)")")"
export CUMM_CUDA_ARCH_LIST="9.0"
export SPCONV_DISABLE_JIT="1"
source /users/cymoser/projects/PostProcess/venv/bin/activate
ulimit -c 0
export OMP_NUM_THREADS=4
export HF_HUB_DISABLE_XET=1
