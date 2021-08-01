import argparse
import os
import numpy as np
import os.path as osp
import warnings
from collections import OrderedDict

import torch
import mmcv
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmpose.models import build_posenet
from mmdet.core import get_classes

try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import wrap_fp16_model

try:
    from brl_graph.utils.inference_mmdet import *
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from brl_graph.utils.inference_mmpose import (
    inference_top_down_pose_model, inference_bottom_up_pose_model, 
    run_human_pose_estimation, init_pose_model, vis_pose_result)
from brl_graph.utils.utils import *
from brl_graph.utils.timer import (
    Timer, cal_total_mean_time)
from brl_graph.utils.parse import InferenceParser
from brl_graph.utils.warmup_gpus import WarmUpGPUs


def main():
    args = InferenceParser()
    args().device = 'cuda:' + args().device
    cfg_path = f'/root/mmpose/brl_graph/models/cfg_list.yaml'
    
    pose_cfg, pose_ckpt = get_base_pose_info(cfg_path, args().model, args().dataset, args().cfgnum)
    
    cfg = Config.fromfile(pose_cfg) 
    pose_cfg_name = osp.splitext(osp.basename(pose_cfg))[0]
    
    pose_model = init_pose_model(
        pose_cfg, 
        pose_ckpt,
        apply_speedup=args().speedup,
        device=args().device.lower())
    
    print(pose_model.cfg.channel_cfg)
    print(pose_model.cfg.model.test_cfg)

    if args().speedup:
        print("apply speedup")
    pose_model_type = pose_model.cfg.model.type
    dataset = pose_model.cfg.data['test']['type']

    case_num = check_case_len(args().case)
    cases = "case_" + case_num # case_00, case_01, ...
    
    if osp.exists(args().img):
        img_data = get_image_list(args().img)
        img_name_lists = [osp.splitext(osp.basename(img_name))[0] for img_name in img_data]
    else:
        raise ImageDataNotLoadedError(
            f'image data is not loaded. check the exist of path or input path value\n'
        )

    if len(img_data) <= 11:
        print("::Image List::")
        print(img_data)
    print(f"Image Len: {len(img_data)}")
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = False
    return_heatmap = True
    output_layer_names = None # e.g. use ('backbone', ) to return backbone feature

    if args().warmup:
        warmup_iter = 10
        warmup = WarmUpGPUs(args().device.lower())
        warmup.simple_warmup(warmup_iter)
    
    # create to save the results of detection and pose estimation
    base_save_path = f'/root/volume/{args().infer_data}/infer_results/' \
    + f'{args().det}_{pose_cfg_name}/{cases}'
  
    # result_save_path = osp.join(base_save_path, 'result_info')
    # mmcv.mkdir_or_exist(result_save_path)
    # result_save_interval = int(len(img_data)/50)
    # print("result interval:", result_save_interval)

    info_ordered_dict = OrderedDict()
    error_idx_ordered_dict = OrderedDict()

    if pose_model_type == 'TopDown':
        print("\n*****************************<Process Start>*****************************")
        
        det_yaml_path = f'/root/mmpose/brl_graph/detectors/cfg_lists.yaml'
        det_cfg, det_ckpt = get_det_info(
            det_yaml_path, args().det, args().det_cfgnum) #format:: task:[cfg, ckpt]

        detection_model = init_detector(
            det_cfg, det_ckpt, device=args().device.lower())

        for idx, img in enumerate(img_data):
            try:
                # detection
                det_results = run_detection(detection_model, img)

                # extraction human bbox
                human_boxes = process_mmdet_results(det_results)

                # pose estimation
                pose_results, heatmap = run_human_pose_estimation(
                                pose_model,
                                img,
                                human_boxes,
                                bbox_thr=args().bbox_thr,
                                format='xyxy',
                                dataset=dataset,
                                return_heatmap=return_heatmap,
                                outputs=output_layer_names)
                #TODO keypoint에 대한 skeleton 예측 결과 그리는 부분 추가
                # save some results
                # visualize results
                if args().det_save:
                    if idx == 0:
                        det_save_path = osp.join(base_save_path, 'vis_det')
                        mmcv.mkdir_or_exist(det_save_path)

                    from  brl_graph.utils.imageio import draw_box_results
                    draw_box_results(
                        img,
                        det_results, 
                        det_save_path,
                        args().bbox_thr,
                        only_person=True)
                
                if args().pose_save:
                    if idx == 0:
                        pose_save_path = osp.join(base_save_path, 'vis_pose')
                        mmcv.mkdir_or_exist(pose_save_path)

                    from  brl_graph.utils.imageio import draw_pose_results_with_box
                    draw_pose_results_with_box(
                        pose_model,
                        img,
                        pose_results,
                        radius=args().radius,
                        thickness=args().radius//2,
                        kpt_score_thr=args().kpt_thr,
                        dataset=dataset,
                        show=args().show,
                        pose_save_path=pose_save_path)
                
            except ValueError as ex:
                print(ex)
                
        

    print("\nAll finished\n")
    print("*"*50); print()
    
    
if __name__ == '__main__':
    main()
