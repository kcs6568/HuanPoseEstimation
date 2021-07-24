#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
IMG=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# echo $CONFIG
# echo $CHECKPOINT
# echo $DATA

python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    $HOME/mmpose/cskim_custom/running/one_time_inference.py $CONFIG $CHECKPOINT $IMG ${@:4}