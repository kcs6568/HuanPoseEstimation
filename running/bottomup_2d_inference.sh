#!/usr/bin/env bash

GPUS=1
CONFIG=$1
CHECKPOINT=$2
IMG=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# echo $CONFIG
# echo $CHECKPOINT
# echo $DATA

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $HOME/mmpose/brl_graph/running/bottomup_2d_inference.py \
    $CONFIG $CHECKPOINT $IMG ${@:4}