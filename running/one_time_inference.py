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
    from cskim_custom.utils.inference_mmdet import *
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from cskim_custom.utils.inference_mmpose import (
    inference_top_down_pose_model, inference_bottom_up_pose_model, 
    run_human_pose_estimation, init_pose_model, vis_pose_result)
from cskim_custom.utils.utils import *
from cskim_custom.utils.timer import (
    Timer, cal_total_mean_time)
from cskim_custom.utils.parse import InferenceParser
from cskim_custom.utils.warmup_gpus import WarmUpGPUs
from cskim_custom.utils.fileio import *


def main():
    args = InferenceParser()
    args().device = 'cuda:' + args().device
    cfg_path = f'/root/mmpose/cskim_custom/models/cfg_list.yaml'
    
    pose_cfg, pose_ckpt = get_base_pose_info(cfg_path, args().model, args().dataset, args().cfgnum)
    
    cfg = Config.fromfile(pose_cfg) 
    pose_cfg_name = osp.splitext(osp.basename(pose_cfg))[0]
    
    crop_boxes = args().h_crop * args().w_crop
    pose_model = init_pose_model(
        pose_cfg, 
        pose_ckpt,
        apply_speedup=args().speedup,
        device=args().device.lower())
    
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

    is_crop = is_cropping(args().resize_scale, args().h_crop, args().w_crop)
    print("Is Crop?:", is_crop)

    if args().warmup:
        warmup_iter = 10
        warmup = WarmUpGPUs(args().device.lower())
        warmup.simple_warmup(warmup_iter)
    
    # create to save the results of detection and pose estimation
    base_save_path = f'/root/volume/{args().infer_data}/infer_results/' \
    + f'{args().det}_{pose_cfg_name}/{cases}/resize_{args().resize_scale}' \
    + f'/H{args().h_crop}_W{args().w_crop}_kpt{args().kpt_thr}_bbox{args().bbox_thr}'

    det_save_path = osp.join(base_save_path, 'vis_det')
    pose_save_path = osp.join(base_save_path, 'vis_pose')
    result_save_path = osp.join(base_save_path, 'result_info')
    
    mmcv.mkdir_or_exist(det_save_path)
    mmcv.mkdir_or_exist(pose_save_path)
    mmcv.mkdir_or_exist(result_save_path)
    

    naming_info = [args().h_crop, args().w_crop, args().kpt_thr, args().bbox_thr]

    result_save_interval = int(len(img_data)/50)
    print("result interval:", result_save_interval)

    info_ordered_dict = OrderedDict()
    error_idx_ordered_dict = OrderedDict()
    det_time_results = []
    pose_time_results = []
    if pose_model_type == 'TopDown':
        print("\n*****************************<Process Start>*****************************")
        
        det_yaml_path = f'/root/mmpose/cskim_custom/detectors/cfg_lists.yaml'
        det_cfg, det_ckpt = get_det_info(
            det_yaml_path, args().det, args().det_cfgnum) #format:: task:[cfg, ckpt]

        detection_model = init_detector(
            det_cfg, det_ckpt, device=args().device.lower())

        for idx, img in enumerate(img_data):
            try:
                # detection
                det_results, det_infer_time = run_detection(detection_model, img)

                # extraction cropped human bbox
                cropped_human_boxes = process_mmdet_results(
                    det_results, 
                    is_crop,
                    args().h_crop, 
                    args().w_crop)

                # pose estimation
                pose_results, heatmap, pose_infer_time = run_human_pose_estimation(
                                pose_model,
                                img,
                                cropped_human_boxes,
                                args().resize_scale,
                                bbox_thr=args().bbox_thr,
                                format='xyxy',
                                dataset=dataset,
                                return_heatmap=return_heatmap,
                                outputs=output_layer_names)

                det_time_results.append(det_infer_time)
                pose_time_results.append(pose_infer_time)


                # save some results
                if idx%result_save_interval == 0:
                    print(f'process {idx}th image({img_name_lists[idx]}) finish')                                     
                    info_ordered_dict[idx] = {
                        'mode': args().mode,
                        'image_path': img,
                        'det_infer_time': round(det_infer_time*(1e-3), 4),
                        'pose_infer_time': round(pose_infer_time*(1e-3), 4),
                        }

                    # visualize results
                    if args().det_save:
                        from  cskim_custom.utils.imageio import draw_box_results
                        draw_box_results(
                            img,
                            det_results, 
                            det_save_path,
                            args().bbox_thr,
                            naming_info=naming_info,
                            only_person=True)
                    
                    if args().pose_save:
                        from  cskim_custom.utils.imageio import draw_pose_results_with_box
                        draw_pose_results_with_box(
                            pose_model,
                            img,
                            pose_results,
                            radius=args().radius,
                            thickness=args().radius//2,
                            kpt_score_thr=args().kpt_thr,
                            dataset=dataset,
                            show=args().show,
                            pose_save_path=pose_save_path,
                            naming_info=naming_info)
                
            except ValueError as ex:
                det_time_results.append(0)
                pose_time_results.append(0)
                error_idx_ordered_dict[idx] = {
                    'mode': args().mode,
                    'image_path': img,
                    'det_infer_time': 0,
                    'pose_infer_time': 0, 
                    'error_type': type(ex).__name__,
                    'error_msg': str(ex)
                }
        

    print("\nAll finished\n")
    print("*"*50); print()
    info_save_path = osp.join(result_save_path, 'result_info.json')
    error_info_save_path = osp.join(result_save_path, 'error_info.json')
    mmcv.dump(info_ordered_dict, info_save_path, file_format='json', sort_keys=False, indent='\t')
    mmcv.dump(error_idx_ordered_dict, error_info_save_path, file_format='json', sort_keys=False, indent='\t')
    
    det_error_count = det_time_results.count(0)
    pose_error_count = pose_time_results.count(0)
    
    if det_error_count != pose_error_count:
        raise ErrorCountingNotMatchError(
            "detection error and pose estimation error is not matched")
    else:
        error_count = det_error_count

    data_len = len(img_data)
    correct_data_len = data_len - error_count
    det_total_time, det_mean_time, pose_total_time, pose_mean_time, all_total_time, all_mean_time = calculate_time(
        data_len, det_time_results, pose_time_results)

    print(f"[The number of all images] --------> {data_len} images")
    print(f"[The number of correct images] ----> {correct_data_len} images\n")
    print("<Detection Information>")
    print(f"[Detection Process Time] ----------> Total: {det_total_time} / Mean: {det_mean_time}s/img\n")
    
    print("<Pose Estimation Information>")
    print(f"[Bounding Box Resizing] -----------> {args().resize_scale}")
    print(f"[Crop Rate] -----------------------> Width: {args().w_crop} / Height: {args().h_crop} (the number of cropped bboxes per bbox: {args().w_crop * args().h_crop})")
    print(f"[Pose Process Time] ----------------> Total: {pose_total_time} / Mean: {pose_mean_time}s/img\n")

    print("<All Inference Time>")   
    print(f"[All Process Time] -------------------> Total: {all_total_time} / Mean: {all_mean_time}s /img\n")

    final_results = OrderedDict()
    final_results['preprocess'] = {
        'total_image': data_len,
        'real_used_image': correct_data_len,
        'bboc_resize_scale': args().resize_scale,
        'h_crop_rate': args().h_crop,
        'w_crop_rate': args().w_crop
    }
    final_results['detection'] = {
        'total_time': det_total_time,
        'mean_time': det_mean_time
    }
    final_results['pose_estimation']={
        'total_time': pose_total_time,
        'mean_time': pose_mean_time
    }
    final_results['all_task'] = {
        'all_total_time': all_total_time,
        'all_mean_time': all_mean_time
    }
    
    final_results_path = osp.join(result_save_path, 'final_results.json')
    mmcv.dump(final_results, final_results_path, file_format='json', sort_keys=False, indent='\t')
    
    print("\n***************really finished***************\n")
    
if __name__ == '__main__':
    main()
