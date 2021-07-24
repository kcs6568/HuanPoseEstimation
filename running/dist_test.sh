#!/usr/bin/env bash

ls

# ${num}: the order of arguments in testing commands
# ./dist_test.sh [CONFIG] [CHECKPOINT] [GPUS] (must be this order in command)
# CONFIG=$1
# CHECKPOINT=$2
# GPUS=$3
# PORT=${PORT:-29500}

# # dirname: the directory path except for filename
# # dirname ${num}: the directory path in the position of {num} argument
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# # ${@:$}: other options not are explicited in the shell script command
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $HOME/mmpose/cskim_custom/test.py $CONFIG $CHECKPOINT $GPUS --launcher pytorch ${@:4}



