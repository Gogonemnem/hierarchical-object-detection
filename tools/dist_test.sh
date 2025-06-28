#!/usr/bin/env bash

# Default script is for standard testing
SCRIPT_PATH="$(dirname "$0")/test.py"
# Array to hold arguments for the python script
PY_ARGS=()

# Check for a --robustness flag to switch scripts
# and filter it out from the arguments passed to python.
for arg in "$@"; do
  if [[ "$arg" == "--robustness" ]]; then
    SCRIPT_PATH="$(dirname "$0")/analysis_tools/test_robustness.py"
  else
    PY_ARGS+=("$arg")
  fi
done

# The python script arguments are now in the PY_ARGS array
CONFIG=${PY_ARGS[0]}
CHECKPOINT=${PY_ARGS[1]}
GPUS=${PY_ARGS[2]}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$SCRIPT_PATH" \
    "$CONFIG" \
    "$CHECKPOINT" \
    --launcher pytorch \
    "${PY_ARGS[@]:3}"