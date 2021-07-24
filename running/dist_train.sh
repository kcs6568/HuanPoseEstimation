#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
DATASET=$3
CFGNUM=$4


PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# ls $(dirname $0)
# echo "$(dirname $1)"
# ls ./$(dirname $1)/..

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $HOME/mmpose/cskim_custom/running/train.py $CONFIG $DATASET $CFGNUM --launcher pytorch ${@:5}
    
    
