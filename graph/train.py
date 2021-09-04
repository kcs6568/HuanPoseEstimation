import enum
import logging
import os
import argparse
import copy
import os.path as osp
from subprocess import check_output
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import mmcv
from mmcv import Config
from mmcv.runner import init_dist, set_random_seed
from mmcv.runner import EpochBasedRunner, DistSamplerSeedHook, OptimizerHook
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel

from mmpose.models import backbones, build_posenet
from mmpose.models.detectors.associative_embedding import AssociativeEmbedding
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.utils import collect_env, get_root_logger
from mmpose.core.distributed_wrapper import DistributedDataParallelWrapper
from mmpose.core import DistEvalHook, EvalHook, build_optimizers

from brl_graph.utils.utils import *
from brl_graph.utils.parser import TrainParser
from brl_graph.graph.iterative_graph import IterativeGraph
from brl_graph.graph.graph_runner import train_model, GraphRunner



def main():
    args = TrainParser()

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
        if args().numgpus != len(visible_devices.split(',')):
            raise GPUNumberNotMatchError(
                'numgpus argement in parser and CUDA_VISIBLE_DEVICES in os.environ are not matched.'
            )
    else:
        visible_devices = f'The number of device: {args().numgpus}'

    cfg_path = '/root/mmpose/brl_graph/models/cfg_list.yaml'
    pose_cfg, _ = get_base_pose_info(cfg_path, args().pose_model, args().dataset, args().cfgnum)
    cfg = Config.fromfile(pose_cfg) 

    pose_cfg_name = osp.splitext(osp.basename(pose_cfg))[0]

    cfg_options={
        'data_root': '/root/data/coco',
        'data.samples_per_gpu': args().samples_per_gpu,
        'data.workers_per_gpu': cfg.data.workers_per_gpu * args().numgpus}
    # if cfg.model.type=='TopDown':
    #     cfg_options['data.val_dataloader.samples_per_gpu'] = args().samples_per_gpu
    #     cfg_options['data.test_dataloader.samples_per_gpu'] = args().samples_per_gpu
    cfg.merge_from_dict(cfg_options)

    print(f'samples: {cfg.data.samples_per_gpu} / workers: {cfg.data.workers_per_gpu}')

    if args().case.isdigit():
        case_num = check_case_len(args().case)
    else:
        case_num = args().case
    case = "/case_" + case_num

    # TODO 저장소 exp 순서대로 저장하는 부분

    if args().resume_from is not None:
        cfg.resume_from = args().resume_from
    if args().no_pret:
        cfg.model.pretrained=None
        print('model pre-trained:', cfg.model.pretrained)
    if args().weights is not None:
        cfg.model.pretrained=args().weights
    if args().autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.graph_optimizer['lr'] = cfg.graph_optimizer['lr'] * len(cfg.gpu_ids) / 8
    if args().no_pret:
        cfg.model.pretrained=None

    if args().launcher == 'none':
        distributed = False   
    else:
        distributed = True
        init_dist(args().launcher, **cfg.dist_params)
    
    if args().gpu_ids is not None:
        cfg.gpu_ids = args().gpu_ids
    else:
        cfg.gpu_ids = range(args().numgpus) if args().gpus is None else range(args().gpus)
    
    work_dir = f'/root/volume/{args().dataset}/train_results/{pose_cfg_name}/{case}/'
    cfg['work_dir'] = work_dir
    mmcv.mkdir_or_exist(osp.abspath(work_dir))
    mmcv.mkdir_or_exist(osp.abspath(work_dir+'ckpts'))
    mmcv.mkdir_or_exist(osp.abspath(work_dir+'logs'))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(work_dir+'logs', f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_cfg_info_dict = dict()
    # env_cfg_info_dict = set_env_cfg_info(args())

    env_info_dict = collect_env(env_cfg_info_dict)
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    ################## start to print environment information
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
    #             dash_line)
    meta['env_info'] = env_info

    # log some basic info
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    logger.info(f'Distributed training: {distributed} --> CUDA_VISIBLE_DEVICES: {visible_devices}')
    logger.info(f'Master Address: {master_addr} / Master Port: {master_port}')
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args().seed is not None:
        logger.info(f'Set random seed to {args().seed}, '
                    f'deterministic: {args().deterministic}')
        set_random_seed(args().seed, deterministic=args().deterministic)
    cfg.seed = args().seed
    meta['seed'] = args().seed

    logger.info("Make Graph Iterative Network")
    num_joints = cfg.model.keypoint_head.num_joints
    model = GraphRunner(cfg.model, num_joints)

    logger.info("Set Data Loader")
    datasets = [build_dataset(cfg.data.train)]
    dataset = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    dataloader_setting = dict(
        samples_per_gpu=cfg.data.get('samples_per_gpu', {}),
        workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))
    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in dataset
    ]
    logger.info(f'DataLoader: {data_loaders}')


    use_adverserial_train = cfg.get('use_adversarial_train', False)
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        if use_adverserial_train:
            # Use DistributedDataParallelWrapper for adversarial training
            backbone = DistributedDataParallelWrapper(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            logger.info('MMDDP Backbone and Model Settings')
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
            logger.info(f"Model Info: {model.__class__.__name__} / {model.__class__}")
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)


    optimizer = build_optimizers(model, cfg.graph_optimizer) 
    logger.info("Program Main Runner Initialization")
    model_runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=work_dir,
        logger=logger,
        meta=meta)
    model_runner.timestamp = timestamp

    if use_adverserial_train:
        # The optimizer step process is included in the train_step function
        # of the model, so the runner should NOT include optimizer hook.
        optimizer_config = None
    else:
        if distributed and 'type' not in cfg.graph_optimizer_config:
            # same as EpochBasedRunner optimizer
            optimizer_config = OptimizerHook(**cfg.graph_optimizer_config)
        else:
            optimizer_config = cfg.graph_optimizer_config

    cfg.checkpoint_config['out_dir'] = f'{work_dir}/ckpts'
    model_runner.register_training_hooks(
        lr_config=cfg.lr_config, # optimizer scheduler
        optimizer_config=optimizer_config, # optimizer information
        checkpoint_config=cfg.checkpoint_config,
        log_config=cfg.log_config)

    if distributed:
        model_runner.register_hook(DistSamplerSeedHook())

    # loading validation dataset
    if not args().no_validate:
        logger.info("Validation Dataloader Setting")
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            # validation sample이 1이 아닐 때 size runtime error 발생
            # RuntimeError: stack expects each tensor to be equal size, but got [428, 640, 3] at entry 0 and [500, 375, 3] at entry 1
            samples_per_gpu=1,
            workers_per_gpu=1,
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            drop_last=False,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        cfg.val_settings = dataloader_setting

        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEvalHook if distributed else EvalHook
        model_runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        model_runner.resume(args().resume_from)
    elif cfg.load_from:
        model_runner.load_checkpoint(cfg.load_from)

    model_runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)
    # cls_loss = nn.CrossEntropyLoss()
    # reg_loss = nn.L1Loss()
    
    # logger.info("Start Model Training")
    # for epoch in range(cfg.total_epochs):
    #     model.train()
    #     losses = torch.tensor(0.)

    #     for iter, data_batch in enumerate(data_loaders[0]):
    #         pred_loc, target_loc = model.train_step(data_batch, optimizer)

    #         for pred, target in zip(pred_loc, target_loc):
    #             losses += reg_loss(pred, target)

    #         # losses.backward()
    #         optimizer.step()

        # train_model(
        #     backbone,
        #     model,
        #     optimizer,
        #     reg_loss,
        #     data_loaders[0]
        # )

        # for iter, batch in enumerate(data_loaders[0]):
        #     losses = torch.tensor(0.)
        #     optimizer.zero_grad()
        #     low_res, high_res = backbone(
        #         batch['img'],
        #         batch['targets'],
        #         batch['masks'],
        #         batch['joints'])

        #     ####################
        #     # data preprocessing
        #     ####################
        #     loc_pred, loc_targets = model.module(batch['targets'][1], high_res)
        #     for pred, target in zip(loc_pred, loc_targets):
        #         losses += reg_loss(pred, target)
        #     # loss = reg_loss(loc_pred, loc_targets)
        #     losses.backward()
        #     optimizer.step()
            
            
        # scheduler.step()
        # print(f"{epoch+1} training clear")

    

if __name__ == '__main__':
    main()