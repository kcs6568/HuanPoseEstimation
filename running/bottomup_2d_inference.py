import argparse
import os
import numpy as np
import os.path as osp
import warnings
from collections import OrderedDict

import torch
import torchvision
import mmcv

from brl_graph.utils.inference_mmpose import (
    inference_bottom_up_pose_model, run_bottomup_hpe,
    init_pose_model, vis_pose_result)
from brl_graph.utils.utils import *
from brl_graph.utils.timer import (
    Timer, cal_total_mean_time)
from brl_graph.utils.parser import BottomUpInferenceParser
from brl_graph.utils.warmup_gpus import WarmUpGPUs


def main():
    args = BottomUpInferenceParser()
    args().device = 'cuda:' + args().device
    cfg_path = f'/root/mmpose/brl_graph/models/cfg_list.yaml'
    
    pose_cfg, pose_ckpt = get_base_pose_info(cfg_path, args().model, args().dataset, args().cfgnum)
    
    pose_cfg_name = osp.splitext(osp.basename(pose_cfg))[0]
    
    pose_model = init_pose_model(
        pose_cfg, 
        pose_ckpt,
        apply_speedup=args().speedup,
        device=args().device.lower())
    assert pose_model.cfg.model.type=='AssociativeEmbedding', (
        "This process is applied to bottomup approach")

    if args().speedup:
        print("apply speedup")
    
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

    return_heatmap = True
    output_layer_names = None # e.g. use ('backbone', ) to return backbone feature

    if args().warmup:
        warmup_iter = 10
        warmup = WarmUpGPUs(args().device.lower())
        warmup.simple_warmup(warmup_iter)
    
    # create to save the results of detection and pose estimation
    base_save_path = f'/root/volume/{args().infer_data}/infer_results/bottomup/' \
    + f'{pose_cfg_name}/{cases}'

    info_ordered_dict = OrderedDict()
    error_idx_ordered_dict = OrderedDict()

    print("\n*****************************<Process Start>*****************************")
    
    for idx, img in enumerate(img_data):
        try:
            # pose estimation
            pose_results, heatmap = run_bottomup_hpe(
                            pose_model,
                            img,
                            pose_nms_thr=args().pose_nms_thr,
                            return_heatmap=return_heatmap,
                            outputs=output_layer_names)

            #TODO keypoint에 대한 skeleton 예측 결과 그리는 부분 추가
            # save some results
            # visualize results
            
            if args().pose_save:
                if idx == 0:
                    pose_save_path = osp.join(base_save_path, 'vis_pose')
                    mmcv.mkdir_or_exist(pose_save_path)

                from  brl_graph.utils.imageio import draw_bottomup_pose_results
                draw_bottomup_pose_results(
                    pose_model,
                    img,
                    pose_results,
                    radius=args().radius,
                    thickness=args().radius//2,
                    dataset=dataset,
                    kpt_score_thr=args().kpt_thr,
                    show=args().show,
                    pose_save_path=pose_save_path)
        except ValueError as ex:
            print(ex)
        

    print("\nAll finished\n")
    print("*"*50); print()
    
    
if __name__ == '__main__':
    main()
