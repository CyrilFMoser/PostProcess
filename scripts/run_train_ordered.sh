#!/bin/bash
# Shortcut: medium_skip, resume newest checkpoint
# Usage: sbatch run_train_ordered.sh
exec "$(dirname "$0")/run_train.sh" model=medium_skip training.chkpt_newest=true "$@"
