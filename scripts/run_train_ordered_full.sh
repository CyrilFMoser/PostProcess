#!/bin/bash
# Shortcut: full_skip, resume newest checkpoint
# Usage: sbatch run_train_ordered_full.sh
exec "$(dirname "$0")/run_train.sh" model=full_skip training.chkpt_newest=true "$@"
