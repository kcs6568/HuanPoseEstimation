import os
import argparse

from mmcv import DictAction


class TrainParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='mmpose test model')
        self.args = self._parse_args(self.parser)

    
    def _parse_args(self, parser):
        ### parser argument must comply with the order of input commands ###
        parser.add_argument('model', help='used model')
        parser.add_argument('dataset', type=str)
        parser.add_argument('cfgnum', type=int)
        parser.add_argument('--case', type=str)
        parser.add_argument('--devices', type=str, default=None, help='the specific gpus not consecutive order')
        ####################################################################

        parser.add_argument('--weights', default=None, type=str, help='set specific weights')
        parser.add_argument('--seed', type=int, default=None, help='random seed')
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--no-pret', action='store_true')
        parser.add_argument('--worker-unit', type=int)
        parser.add_argument('--samples', type=int, default=32)

        parser.add_argument(
            '--work-dir',
            action='store_false',
            help='the dir to save logs and models')
        parser.add_argument(
            '--resume-from', help='the checkpoint file to resume from')
        parser.add_argument(
            '--no-validate',
            action='store_true',
            help='whether not to evaluate the checkpoint during training')
        group_gpus = parser.add_mutually_exclusive_group()
        group_gpus.add_argument(
            '--gpus',
            type=int,
            help='number of gpus to use '
            '(only applicable to non-distributed training)')
        group_gpus.add_argument(
            '--gpu-ids',
            type=int,
            nargs='+',
            help='ids of gpus to use '
            '(only applicable to non-distributed training)')
        parser.add_argument(
            '--deterministic',
            action='store_true',
            help='whether to set deterministic options for CUDNN backend.')
        parser.add_argument(
            '--cfg-options',
            nargs='+',
            action=DictAction,
            default={},
            help='override some settings in the used config, the key-value pair '
            'in xxx=yyy format will be merged into config file. For example, '
            "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
        parser.add_argument(
            '--launcher',
            choices=['none', 'pytorch', 'slurm', 'mpi'],
            default='none',
            help='job launcher')
        parser.add_argument(
            '--autoscale-lr',
            action='store_true',
            help='automatically scale lr with the number of gpus')

        args = parser.parse_args()
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(args.local_rank)
        
        return args


    def __call__(self):
        return self.args


class TestParser():
    pass


class InferenceParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='mmpose test model')
        self.args = self._parse_args(self.parser)

    
    def _parse_args(self, parser):
        ### self.parser argument must comply with the order of input commands ###
        parser.add_argument('model', help='used model')
        parser.add_argument('cfgnum', type=int)
        parser.add_argument('det', type=str, default=None, help='choice detector the one or all')
        parser.add_argument('det_cfgnum', type=int, default=1)
        parser.add_argument('dataset', type=str)
        parser.add_argument('case', type=str)
        parser.add_argument('img', type=str)
        ####################################################################
        parser.add_argument('--det-save', action='store_true')
        parser.add_argument('--pose-save', action='store_true')
        parser.add_argument('--det-num', type=int, default=1)
        parser.add_argument('--img_root', default='/root/mmpose/test/data/mpii') # original image data directory
        parser.add_argument('--out', help='output result file')
        parser.add_argument('--tovolume', action='store_true')
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--device', type=str, default='0')
        parser.add_argument('--radius', type=int, default=6)
        parser.add_argument('--thick', type=int)
        parser.add_argument('--caption-name', action='store_true',
        default="", help='add model config caption in the out file name')
        parser.add_argument('--debug', action='store_true', help='if true, start debugging in terminal')
        parser.add_argument('--infer-data', default='coco', type=str, help='sets dataset to infer')
        parser.add_argument('--time-sync', action='store_true')
        parser.add_argument('--warmup', action='store_true')
        parser.add_argument('--h-crop', type=int, default=1)
        parser.add_argument('--w-crop', type=int, default=1)
        parser.add_argument('--resize-scale', type=str, default='origin')
        parser.add_argument('--speedup', action='store_false', help='flip false and post process default in cfg')
        parser.add_argument('--mode', type=str, default='inference')
        
        parser.add_argument(
            '--work-dir',
            default='store_false',
            help='the dir to save logs and models')
        parser.add_argument(
            '--fuse-conv-bn',
            action='store_true',
            help='Whether to fuse conv and bn, this will slightly increase'
            'the inference speed')
        parser.add_argument(
            '--det-cat-id',
            type=int,
            default=1,
            help='Category id for bounding box detection model')
        parser.add_argument(
            '--bbox-thr',
            type=float,
            default=0.8,
            help='Bounding box score threshold')
        parser.add_argument(
            '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
        parser.add_argument(
            '--out-img-root',
            type=str,
            default='',
            help='root of the output img file. '
            'Default not saving the visualization images.')
        parser.add_argument(
            '--show',
            action='store_true',
            default=False,
            help='whether to show img')

        args = parser.parse_args()
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(args.local_rank)

        return args


    def __call__(self):
        return self.args