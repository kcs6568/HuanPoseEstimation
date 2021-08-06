#!/usr/bin/env bash

POSE_CFG=$1
POSE_CFGNUM=$2
# DET_CFG=$4
# DET_CFGNUM=$5
CASE=$3
DATASET=$4
GPUS_NUM=$5
GPU_LIST=$6


PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# ls $(dirname $0)
# echo "$(dirname $1)"
# ls ./$(dirname $1)/..

#TODO multi-gpu 설정하는 코드 확인 및 수정
python -m torch.distributed.launch --nproc_per_node=$GPUS_NUM --master_port=$PORT \
    $HOME/mmpose/brl_graph/running/train.py \
    $POSE_CFG $POSE_CFGNUM $CASE $DATASET $GPU_LIST\
    --launcher pytorch ${@:5}
    
    
