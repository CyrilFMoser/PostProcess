#!/bin/bash
# Shortcut: larger_skip, resume newest checkpoint
# Usage: sbatch run_train_ordered_larger.sh
exec "$(dirname "$0")/run_train.sh" model=larger_skip training.chkpt_newest=true "$@"
