#!/bin/bash
# Bottleneck class agg + class agg at fine resolution.
# Usage: ./run_refine_bottleneck_class.sh [--fresh] <num_jobs> [extra_hydra_overrides...]

DIR="$(cd "$(dirname "$0")" && pwd)"

FRESH_ARG=""
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--fresh" ]; then
        FRESH_ARG="--fresh"
    else
        ARGS+=("$arg")
    fi
done

if [ "${#ARGS[@]}" -lt 1 ]; then
    echo "Usage: $0 [--fresh] <num_jobs> [extra_hydra_overrides...]"
    exit 1
fi

NUM_JOBS=${ARGS[0]}
EXTRA=("${ARGS[@]:1}")

exec "$DIR/run_jobs.sh" $FRESH_ARG "$DIR/run_train_multi.sh" "$NUM_JOBS" \
    model=refine_bottleneck_class \
    "${EXTRA[@]}"
