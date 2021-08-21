import os
import os.path as osp

from mmcv import Config
from brl_graph.utils.utils import dump_yaml


def dump_hyp(args, cfg):
    hyp = dict()
    hyp['SAVE_PATH'], hyp['TRAINING'], hyp['VALIDATION'], hyp['OPTIMIZER'], hyp['ENVIRONMENT'] =\
         dict(), dict(), dict(), dict(), dict()
    
    save_path = hyp['SAVE_PATH']
    train = hyp['TRAINING']
    val = hyp['VALIDATION']
    opt = hyp['OPTIMIZER']
    env = hyp['ENVIRONMENT']

    save_path['work_dir'] = cfg.work_dir
    save_path['log_path'] = osp.join(cfg.work_dir, 'logs')
    save_path['ckpt_path'] = cfg.checkpoint_config['out_dir']
    # save_path['results_path'] = osp.join(cfg.work_dir, 'results')
    
    train['train_samples_per_gpu'] = cfg.data.samples_per_gpu
    train['train_workers_per_gpu'] = cfg.data.workers_per_gpu
    train['input_size'] = cfg.data_cfg.image_size
    train['heatmap_size'] = cfg.data_cfg.heatmap_size
    train['data_type'] = args.dataset
    train['num_gpus'] = args.numgpus
    train['total_epochs'] = cfg.total_epochs

    val['val_samples_per_gpu'] = cfg.val_settings['samples_per_gpu']
    val['val_workers_per_gpu'] = cfg.val_settings['workers_per_gpu']
    val['val_num_gpus'] = cfg.val_settings['num_gpus']
    val['val_distributed'] = cfg.val_settings['dist']
    val['val_drop_last'] = cfg.val_settings['drop_last']
    val['val_shuffle'] = cfg.val_settings['shuffle']
    
    opt['optimizer'] = cfg.optimizer.type
    opt['init_lr'] = cfg.optimizer.lr
    opt['grad_clip'] = cfg.optimizer_config.dictitems

    tmp = dict()
    for i, j in cfg.lr_config.items():
        tmp[i] = j
    opt['lr_strategy'] = tmp
    # opt['lr_strategy'] = {
    #     'policy': cfg.lr_config.policy,
    #     'warmup': cfg.lr_config.warmup,
    #     'warmup_iters': cfg.lr_config.warmup_iters,
    #     'warmup_ratio': cfg.lr_config.warmup_ratio,
    #     'step': cfg.lr_config.step
    # }
    

    env['MASTER_ADDR'] = os.environ['MASTER_ADDR']
    env['MASTER_PORT'] = os.environ['MASTER_PORT']
    env['WORLD_SIZE'] = os.environ['WORLD_SIZE']
    env['RANK'] = os.environ['RANK']
    env['LOCAL_RANK'] = os.environ['LOCAL_RANK']
    env['OMP_NUM_THREADS'] = os.environ['OMP_NUM_THREADS']
    env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']

    hyp_name = 'hyp2.yaml'
    work_path = osp.join(save_path['work_dir'], hyp_name)
    dump_yaml(work_path, hyp)

