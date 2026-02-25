#!/bin/bash
# Shortcut: medium_skip multi-node, resume newest checkpoint
# Usage: sbatch run_train_medium_multi.sh
exec "$(dirname "$0")/run_train_multi.sh" model=medium_skip training.chkpt_newest=true "$@"
