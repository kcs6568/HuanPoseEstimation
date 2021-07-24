import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import init_dist, set_random_seed
from mmcv.utils import get_git_hash

from mmpose import __version__
from mmpose.apis import train_model
from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
from mmpose.utils import collect_env, get_root_logger

from cskim_custom.utils.utils import *
from cskim_custom.utils.parse import TrainParser


def set_env_cfg_info(args):
    env_cfg_info_dict = {
        'Info. List': '     ',
        'Training Model': args.model,
        'Dataset': args.dataset,
        'Config Number': args.cfgnum,
        'Specific-GPUs?': args.devices,
        'Training Case': args.case,
        'Is no pretrained?': args.no_pret,
        'Working_Dir': args.work_dir,
        'GPUs(at the single GPU)': args.gpus,
        'GPU-IDs(at the single GPU)': args.gpu_ids,
        'Deterministic': args.deterministic,
        'Config_Options': args.cfg_options,
        'Launcher': args.launcher,
        'Autoscale_LR': args.autoscale_lr,
        'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
        'Local Rank': os.environ['LOCAL_RANK'],
        'Rank': os.environ['RANK']
    }

    arg_info = '\n'.join([(f'{k}: {v}') for k, v in env_cfg_info_dict.items()])

    return arg_info

def main():
    args = TrainParser()

    cfg_path = '/root/mmpose/cskim_custom/models/cfg_list.yaml'
    
    pose_cfg, pose_ckpt = get_base_pose_info(cfg_path, args().model, args().dataset, args().cfgnum)
    abs_pose_cfg, abs_pose_ckpt = make_base_path_for_pose(args().model, args().dataset, pose_cfg, pose_ckpt)

    cfg = Config.fromfile(abs_pose_cfg) 
    cfgfile_name = osp.splitext(osp.basename(abs_pose_cfg))[0]

    
    
    if args().cfg_options is not None:
        cfg.merge_from_dict(args().cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    case_num = check_case_len(args().case)
    case = "/case_" + case_num
    if args().work_dir:
        cfg.work_dir = "/root/mmpose/cskim_custom/" + args().dataset + \
        "/results/" + cfgfile_name + "/" + case + "/train_results"

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

    # TODO Distributed Multi-gpu training 환경 세팅하기
    if args().devices is not None:
        set_specific_gpus(args().devices)

    if args().no_pret:
        cfg.model.pretrained=None
        print('model pre-trained:', cfg.model.pretrained)
    
    if args().weights is not None:
        cfg.model.pretrained=args.weights

    if args().launcher == 'none':
        distributed = False
    else:
        distributed = True
        # print("launcher:", args().launcher)
        init_dist(args().launcher, **cfg.dist_params)

    # cfg.data.samples_per_gpu = args.samples


    cfg.data.samples_per_gpu = args().samples
    # args().worker_unit = args().devices // 2
    # cfg.data.workers_per_gpu = args().worker_unit * args().devices
    len_devices = len(args().devices.split(","))
    cfg.data.workers_per_gpu *= len_devices

    # print("dist:", distributed)
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
    env_cfg_info_dict = dict()
    env_cfg_info_dict = set_env_cfg_info(args())

    env_info_dict = collect_env(env_cfg_info_dict)
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    ################## start to print environment information
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


if __name__ == '__main__':
    main()
