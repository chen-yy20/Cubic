#!/bin/bash
set -x

SCRIPT=$1
shift || true

if [ -z "$SCRIPT" ]; then
    echo "Usage: bash pysrun.sh <script.py> [script args...]"
    exit 1
fi

if [ ! -f infer.sh ]; then
    echo "infer.sh not found in current directory: $(pwd)"
    exit 1
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)
export NNODES=${NNODES:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-1}
export MASTER_PORT=$(expr $RANDOM % 10000 + 50000)

if [ -z "$PYTHON_EXECUTABLE" ]; then
    if [ -x "$(pwd)/.venv/bin/python" ]; then
        export PYTHON_EXECUTABLE="$(pwd)/.venv/bin/python"
    else
        export PYTHON_EXECUTABLE=$(command -v python || command -v python3)
    fi
fi

if [ -z "$PYTHON_EXECUTABLE" ]; then
    echo "Neither python nor python3 found in PATH"
    exit 1
fi

srun \
    -p "${PARTITION:-debug}" \
    -K \
    -N "${NNODES:-1}" \
    --job-name="${JOB_NAME:-Jano}" \
    --ntasks-per-node="${GPUS_PER_NODE:-1}" \
    --gres=gpu:"${GPUS_PER_NODE:-1}" \
    --export=ALL,MASTER_PORT="$MASTER_PORT" \
    bash infer.sh "$SCRIPT" "$@"
