#!/usr/bin/env bash

POSE_CFG=$1
POSE_CFGNUM=$2
DATASET=$3
CASE=$4
GPUS_NUM=$5
# PORT=$6
PORT=${PORT:-29500}
# PORT0=${PORT:-6000}
# PORT1=${PORT:-6001}
# PORT2=${PORT:-6002}
# PORT3=${PORT:-6003}
# PORT4=${PORT:-6004}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

python -m torch.distributed.launch --nproc_per_node=$GPUS_NUM --master_port=$PORT \
    $(dirname "$0")/custom_train.py $POSE_CFG $POSE_CFGNUM $DATASET $CASE $GPUS_NUM --launcher pytorch ${@:6}
    
    
