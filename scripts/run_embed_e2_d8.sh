#!/bin/bash
# Convenience wrapper for larger_embed with embedding_dim=2, d_head=8.
# embed_ch = 2 * class_subset (e.g. 104 for 52 classes).
# Linear(2, 8) upscale; d_head=8 is the flash-attn minimum.
# Usage: ./run_embed_e2_d8.sh [--fresh] <num_jobs> [extra_hydra_overrides...]

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
    model=larger_embed \
    model.embedding_dim=2 \
    'model.dec_channels_rest=[104]' \
    model.neighbor_transformer.d_head=8 \
    "${EXTRA[@]}"
