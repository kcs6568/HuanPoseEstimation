import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

from mmpose import __version__
from mmpose.apis import train_model
from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
from mmpose.utils import collect_env, get_root_logger

from brl_graph.utils.utils import *
from brl_graph.utils.parser import TrainParser

def main():
    args = TrainParser()

    cfg_path = '/root/mmpose/brl_graph/models/cfg_list.yaml'
    pose_cfg, _ = get_base_pose_info(cfg_path, args().pose_model, args().dataset, args().cfgnum)

    cfg = Config.fromfile(pose_cfg) 
    pose_cfg_name = osp.splitext(osp.basename(pose_cfg))[0]

    if args().cfg_options is not None:
        cfg.merge_from_dict(args().cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    case_num = check_case_len(args().case)
    case = "/case_" + case_num

    if args().resume_from is not None:
        cfg.resume_from = args().resume_from
    if args().gpu_ids is not None:
        cfg.gpu_ids = args().gpu_ids
    else:
        cfg.gpu_ids = range(1) if args().gpus is None else range(args().gpus)

    if args().autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args().launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args().launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args().seed is not None:
        logger.info(f'Set random seed to {args().seed}, '
                    f'deterministic: {args().deterministic}')
        set_random_seed(args().seed, deterministic=args().deterministic)
    cfg.seed = args().seed
    meta['seed'] = args().seed

    model = build_posenet(cfg.model)
    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # save mmpose version, config file content
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmpose_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text,
        )
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args().no_validate),
        timestamp=timestamp,
        meta=meta)