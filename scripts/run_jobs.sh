#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <job_script> <total_runs> [optional_job_arguments...]"
    exit 1
fi

JOB_SCRIPT=$1
NUM_JOBS=$2
shift 2  # Remove the first two args so only the job arguments remain
JOB_ARGS=$@

# Use run_tag as job name if present, otherwise fall back to script name
RUN_TAG=$(echo "$JOB_ARGS" | grep -oP '(?<=run_tag=)\S+')
JOB_NAME=${RUN_TAG:-$(basename $JOB_SCRIPT .sh)}

# Submit the first job and capture the Job ID
FIRST_JOB=$(sbatch --parsable --job-name=$JOB_NAME $JOB_SCRIPT $JOB_ARGS)
if [ $? -ne 0 ]; then
    echo "Error: Initial job submission failed."
    exit 1
fi
echo "Submitted first job: $FIRST_JOB"

# Submit the 2nd, 3rd, ... nth jobs, each depending on the previous one
PREV_JOB=$FIRST_JOB

for (( i=2; i<=$NUM_JOBS; i++ )); do
    # Job starts only if the previous finishes successfully
    NEXT_JOB=$(sbatch --parsable --job-name=$JOB_NAME --dependency=afterany:$PREV_JOB $JOB_SCRIPT $JOB_ARGS)
    echo "Submitted job $i ($NEXT_JOB) - waiting for $PREV_JOB"
    PREV_JOB=$NEXT_JOB
done

echo "----------------------------------------------------"
echo "All jobs submitted."