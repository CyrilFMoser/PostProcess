#!/bin/bash

FRESH=0
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--fresh" ]; then
        FRESH=1
    else
        ARGS+=("$arg")
    fi
done

if [ "${#ARGS[@]}" -lt 2 ]; then
    echo "Usage: $0 [--fresh] <job_script> <total_runs> [optional_job_arguments...]"
    echo "  --fresh  Do not load any checkpoint on the first job (start from scratch)."
    echo "           All subsequent jobs always resume via training.chkpt_newest=true."
    exit 1
fi

JOB_SCRIPT=${ARGS[0]}
NUM_JOBS=${ARGS[1]}
JOB_ARGS=("${ARGS[@]:2}")

# Use run_tag as job name if present, otherwise fall back to script name
RUN_TAG=$(echo "${JOB_ARGS[*]}" | grep -oP '(?<=run_tag=)\S+')
JOB_NAME=${RUN_TAG:-$(basename $JOB_SCRIPT .sh)}

# First job: skip checkpoint loading if --fresh, otherwise resume newest
if [ "$FRESH" -eq 1 ]; then
    FIRST_EXTRA="training.chkpt_newest=false training.load_chkpt=false"
else
    FIRST_EXTRA="training.chkpt_newest=true"
fi

FIRST_JOB=$(sbatch --parsable --job-name=$JOB_NAME $JOB_SCRIPT "${JOB_ARGS[@]}" $FIRST_EXTRA)
if [ $? -ne 0 ]; then
    echo "Error: Initial job submission failed."
    exit 1
fi
echo "Submitted first job: $FIRST_JOB (fresh=$FRESH)"

# All subsequent jobs always resume from the newest checkpoint
PREV_JOB=$FIRST_JOB

for (( i=2; i<=$NUM_JOBS; i++ )); do
    NEXT_JOB=$(sbatch --parsable --job-name=$JOB_NAME --dependency=afterany:$PREV_JOB $JOB_SCRIPT "${JOB_ARGS[@]}" training.chkpt_newest=true)
    echo "Submitted job $i ($NEXT_JOB) - waiting for $PREV_JOB"
    PREV_JOB=$NEXT_JOB
done

echo "----------------------------------------------------"
echo "All jobs submitted."