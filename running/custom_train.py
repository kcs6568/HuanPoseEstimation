import logging
import os
import argparse
import copy
import os.path as osp
import time
import torch

# import torch.optim as optim

import mmcv
from mmcv import Config
from mmcv.runner import init_dist, set_random_seed
from mmcv.utils import get_git_hash
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner import EpochBasedRunner, DistSamplerSeedHook

from mmpose import __version__
from mmpose.apis import train_model
from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
from mmpose.utils import collect_env, get_root_logger
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.core import DistEvalHook, EvalHook, build_optimizers
from mmpose.core.distributed_wrapper import DistributedDataParallelWrapper

from brl_graph.utils.utils import *
from brl_graph.utils.parser import TrainParser
# from brl_graph.utils.save import dump_hyp


def set_env_cfg_info(args):
    env_cfg_info_dict = {
        'Info. List': '     ',
        'Training Model': args.pose_model,
        'Dataset': args.dataset,
        'Config Number': args.cfgnum,
        'Specific-GPUs?': args.devices,
        'Training Case': args.case,
        'Is no pretrained?': args.no_pret,
        'GPUs(at the single GPU)': args.gpus,
        'GPU-IDs(at the single GPU)': args.gpu_ids,
        'Deterministic': args.deterministic,
        'Config_Options': args.cfg_options,
        'Launcher': args.launcher,
        'Autoscale_LR': args.autoscale_lr,
        'Local Rank': os.environ['LOCAL_RANK'],
        'Rank': os.environ['RANK']
    }

    arg_info = '\n'.join([(f'{k}: {v}') for k, v in env_cfg_info_dict.items()])

    return arg_info


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
        'data.samples_per_gpu': args().samples_per_gpu}
    if cfg.model.type=='TopDown':
        cfg_options['data.val_dataloader.samples_per_gpu'] = args().samples_per_gpu
        cfg_options['data.test_dataloader.samples_per_gpu'] = args().samples_per_gpu
    cfg.merge_from_dict(cfg_options)

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
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8
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
    env_cfg_info_dict = set_env_cfg_info(args())

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

    model = build_posenet(cfg.model)
    datasets = [build_dataset(cfg.data.train)]
    
    # prepare data loaders
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
    # mmpose.datasets.datasets.bottom_up.bottom_up_coco.py->BottomUpCocoDataset
    logger.info("Set Data Loader")
    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in dataset
    ]
    logger.info(f'DataLoader: {data_loaders}')

    use_adverserial_train = cfg.get('use_adversarial_train', False)
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        if use_adverserial_train:
            # Use DistributedDataParallelWrapper for adversarial training
            model = DistributedDataParallelWrapper(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            logger.info('MMDDP Settings')
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
            logger.info(f"Model Info: {model.__class__.__name__} / {model.__class__}")
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
   

    # build runner
    optimizer = build_optimizers(model, cfg.optimizer) 

    logger.info("Program Main Runner Initialization")
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=work_dir,
        logger=logger,
        meta=meta)
    runner.timestamp = timestamp

    if use_adverserial_train:
        # The optimizer step process is included in the train_step function
        # of the model, so the runner should NOT include optimizer hook.
        cfg.optimizer_config = None

    cfg.checkpoint_config['out_dir'] = f'{work_dir}/ckpts'
    runner.register_training_hooks(
        lr_config=cfg.lr_config, # optimizer scheduler
        optimizer_config=cfg.optimizer_config, # optimizer information
        checkpoint_config=cfg.checkpoint_config,
        log_config=cfg.log_config)

    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # loading validation dataset
    if not args().no_validate:
        logger.info("Validation Dataloader Setting")
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            samples_per_gpu=16,
            workers_per_gpu=2,
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
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    print(cfg.lr_config.items())
    print(type(cfg.lr_config.items()))
    if cfg.resume_from:
        runner.resume(args().resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    #TODO hyp 저장 코드 수정
    # if args().dump_hyp:
    #     dump_hyp(args(), cfg)
    #     exit()

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
    # total_loss = runner.get_total_loss()


if __name__ == '__main__':
    main()