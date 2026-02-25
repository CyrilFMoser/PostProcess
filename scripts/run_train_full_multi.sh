#!/bin/bash
# Shortcut: full_skip multi-node, resume newest checkpoint
# Usage: sbatch run_train_full_multi.sh
exec "$(dirname "$0")/run_train_multi.sh" model=full_skip training.chkpt_newest=true "$@"
