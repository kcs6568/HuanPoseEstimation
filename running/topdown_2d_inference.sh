#!/usr/bin/env bash

GPUS=1
POSE_CFG=$1
POSE_CKPT=$2
DET_CFG=$3
DET_CKPT=$4
IMG=$5
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# echo $CONFIG
# echo $CHECKPOINT
# echo $DATA

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $HOME/mmpose/brl_graph/running/topdown_2d_inference.py \
    $POSE_CFG $POSE_CKPT $DET_CFG $DET_CKPT $IMG ${@:4}