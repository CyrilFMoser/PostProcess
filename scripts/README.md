# Resume medium_skip
sbatch scripts/run_train.sh model=medium_skip training.chkpt_newest=true

# New experiment: different lr with a name tag
sbatch scripts/run_train.sh model=medium_skip training.lr=1e-4 run_tag=lowlr

# Full skip, multi-node
sbatch scripts/run_train_multi.sh model=full_skip training.chkpt_newest=true

# Just validate
sbatch scripts/run_train.sh model=medium_skip training.chkpt_newest=true training.validate_only=true
