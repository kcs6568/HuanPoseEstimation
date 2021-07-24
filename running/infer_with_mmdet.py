import argparse
import os
import numpy as np
import os.path as osp
import warnings

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
    from cskim_custom.utils.inference_mmdet import (
        inference_detector, init_detector, process_mmdet_results, show_result)
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from cskim_custom.utils.inference_mmpose import (
    inference_top_down_pose_model, inference_bottom_up_pose_model, init_pose_model, vis_pose_result)
from cskim_custom.utils.utils import *
from cskim_custom.utils.timer import (
    Timer, cal_total_mean_time)
from cskim_custom.utils.parse import InferenceParser
from cskim_custom.utils.warmup_gpus import WarmUpGPUs


def main():
    print("\n" + ShadeColor.WHITE + TextColor.BLACK + "Start pose inference with MMDetection" + TextColor.RESET + "\n\n")
    
    exp_info_dict = dict()
    
    args = InferenceParser()
    args().device = 'cuda:' + args().device
    cfg_path = f'/root/mmpose/cskim_custom/models/cfg_list.yaml'
    
    pose_cfg, pose_ckpt = get_base_pose_info(cfg_path, args().model, args().dataset, args().cfgnum)
    abs_pose_cfg, abs_pose_ckpt = make_base_path_for_pose(
        args().model, 
        args().dataset, 
        pose_cfg, 
        pose_ckpt)

    cfg = Config.fromfile(abs_pose_cfg) 

    # print(cfg.)
    
    cfgfile_name = osp.splitext(osp.basename(abs_pose_cfg))[0]

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = False

    case_num = check_case_len(args().case)
    cases = "case_" + case_num # case_00, case_01, ...

    base_dir = f"/root/volume/{args().infer_data}/infer_results/{args().model}/{cfgfile_name}/(det model name)/results/"
    
    
    pose_model_type = cfg.model.type
    print(TextColor.YELLOW + "loading " + TextColor.MAGENTA + "image data for inference... " + TextColor.RESET, end="  ")
    img_data = get_image_list(args().img)
    print(ShadeColor.BRIGHT_BLUE + "complete!" + ShadeColor.RESET + "\n")
    
    if len(img_data) > 1:
        is_batch = True
    else:
        is_batch = False # if one image, img_data is str

    print("Image List::::")
    print(img_data, "\n")
    
    img_name_lists = [osp.splitext(osp.basename(img_name))[0] for img_name in img_data]

    # start bbox inference
    print(TextColor.CYAN + "start detection the bounding boxes" + TextColor.RESET + "\n") 


    print("*"*40)

    # optional
    return_heatmap = True

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    dict_pose_time_results = dict()
    dict_pose_results = dict()

    if args().warmup:
        print(ShadeColor.BLUE + TextColor.WHITE + "Apply WarmUp Inference Task" + ShadeColor.RESET + "\n\n")
        # Warm up GPUs for initialization
        warmup_iter = 10
        warmup = WarmUpGPUs(args().device.lower())
        warmup.simple_warmup(warmup_iter)
    else:
        print(ShadeColor.RED + TextColor.WHITE + "No WarmUp Inference Task" + ShadeColor.RESET + "\n\n")

    # exit()

    print(args().det_save)

    if pose_model_type == 'TopDown':
        det_yaml_path = f'/root/mmpose/cskim_custom/detectors/cfg_lists.yaml'
        det_dict = get_base_det_info(det_yaml_path, args().det, args().det_num) #format:: task:[cfg, ckpt]

        for task, info in det_dict.items():
            print(f'Task: {task}')
            print(f'Information: {info[0]} / {info[1]}')

        det_models = dict()

        for task, data in det_dict.items():
            det_model = init_detector(data[0], data[1], device=args().device.lower())
            det_models[task] = det_model
            # det_models[task] = det_model

        det_time_results = dict()
        det_results = dict()
        # print(f"{task.upper()} Time Checking... ", end="")
# print("Finish!")
                    # print("Wait other process for synchronization...")


        for task, det_model in det_models.items():
            tmp_time = []    
            tmp_det = []
            print("*"*90)
            print(f'Current Task: {task}\n')
            if args().time_sync:
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)

                for idx, img in enumerate(img_data):

                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)

                    start.record()
                    mmdet_results = inference_detector(det_model, img)
                    end.record()
                  
                    torch.cuda.synchronize()                   
                    time_sec = round(start.elapsed_time(end) * (1e-3), 3)

                    tmp_time.append(time_sec)
                    print(f'--> Object detector inference time: {time_sec}\n')
                    

                    # if len(np.shape(mmdet_results)) == 1:
                    #     tmp_det.append(mmdet_results)
                    #     # print("One image reulst saved")
                    #     # print(np.shape(tmp_det))
                    #     # print(np.shape(tmp_det[0]))

                    # elif len(np.shape(mmdet_results)) == 2:
                    #     tmp_det = mmdet_results

                    # print(start.elapsed_time(end))

            else:
                for idx, img in enumerate(img_data):
                    with Timer(
                    print_msg='Object detector inference time',
                    is_batch=is_batch,
                    model_type=pose_model_type,
                    det_task=task,
                    data_len=len(img_data)) as t:
                        t.start()
                        mmdet_results = inference_detector(det_model, img)
                        t.finish()
                        # det_time_results[task] = t.get_time()

                        # print(np.shape(mmdet_results))
                        # print(len(np.shape(mmdet_results)))
                        # print(len(np.shape(mmdet_results[0])))

                        tmp_time.append(t.get_time())
                        # print("-"*50)
                        if len(np.shape(mmdet_results)) == 1:
                            tmp_det.append(mmdet_results)
                            # print("One image reulst saved")
                            # print(np.shape(tmp_det))
                            # print(np.shape(tmp_det[0]))

                        elif len(np.shape(mmdet_results)) == 2:
                            tmp_det = mmdet_results

                    
            det_time_results[task] = tmp_time # time results for each task
            det_results[task] = tmp_det # inference results for each task
            # print(det_results.keys())
            # print(np.shape(list(det_results.values())))
            # print(np.shape(det_results['faster_rcnn'][0]))

        
        # print(det_results.keys())
        # print(np.shape(list(det_results.values())))

        # for i, j in det_results.items():
        #     print(i, np.shape(j), np.shape(j[0]))

        # exit()

        # for task, time in det_time_results.items():
        #     print(f'{task.upper()}: {time} / Std results: {np.std(time)}')
        
        print("\n")
        det_total_results = cal_total_mean_time(det_time_results)

        # memo_root = '/root/mmpose/cskim_custom/test/results.txt'
        # with open(memo_root, 'a') as memo:
        #     memo.write(f'cfg info: {pose_cfg} / {args().cfgnum}\n')
        #     memo.write(f'sync: {args().time_sync} / warmup: {args().warmup}\n')
        #     for task, task_results in results.items():
        #         data = f'{task.upper()}: {task_results}\n' 
        #         print(data)
        #         memo.write(data)

        #     memo.write('*'*40 + "\n")

        # print(results)

        # exit()

        print(ShadeColor.BRIGHT_BLUE + "complete!" + TextColor.RESET + "\n")
        
        if args().det_save: 
            ####################################### INFO ###########################################
            # if np.shape(mmdet_results)의 값이 80이라면, 이는 COCO dataset으로 학습이 되었기 때문에,
            # 80개의 클래스에 대한 예측 값을 출력하는 것임
            ########################################################################################
            from  cskim_custom.utils.imageio import save_box_results

            print(TextColor.CYAN + "saving bbox prediction for each detector... " + TextColor.RESET, end="") 
            save_box_results(det_results, img_data, args().time_sync, args().warmup, args().case, is_batch)
            print(ShadeColor.BRIGHT_BLUE + "complete!" + TextColor.RESET + "\n")


        print(TextColor.CYAN + "getting human bbox prediction in detection results... " + TextColor.RESET, end="") 
        
        # extract only human box excluding other objects
        det_person_results = dict()

        for task, box_results in det_results.items():
            det_person_results[task]=process_mmdet_results(box_results, args().det_cat_id)
        
        print(ShadeColor.BRIGHT_BLUE + "complete!" + TextColor.RESET + "\n")

############################################################################################################################################

        # start pose inference
        print(TextColor.YELLOW + "Start human pose inference... " + ShadeColor.RESET, end="")



        print(TextColor.YELLOW + "create " + TextColor.MAGENTA + "pose model.... " + TextColor.RESET, end="  ")
        pose_model = init_pose_model(
            abs_pose_cfg, abs_pose_ckpt, device=args().device.lower()) # build the model and load checkpoint
        print(ShadeColor.BRIGHT_BLUE + "complete!" + TextColor.RESET  + "\n")
      
        datasets = pose_model.cfg.data['test']['type']

        print("Pose Estimation Model:", pose_model_type.upper())
        print("Time Sync:", args().time_sync)
        print("Warm-Up:", args().warmup, "\n")
        
        for task, human_box_results in det_person_results.items(): # 10개 이미지
            print("Start Top-down inferencing\n")
            tmp_time_per_img = []
            tmp_pose_result_per_img = []
            # pose_results, returned_outputs = [], []
            
            # print(human_box_results)
            
            if args().time_sync:
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)

                for idx, human_boxes in enumerate(human_box_results):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    print(f"Pose Estimation Time Checking on {task.upper()} Human Detection Results... ", end="")
                    start.record()
                    pose_result_per_img, heatmap_per_img = inference_top_down_pose_model(
                            pose_model,
                            img_data[idx],
                            human_boxes,
                            bbox_thr=args().bbox_thr,
                            format='xyxy',
                            dataset=datasets,
                            return_heatmap=return_heatmap,
                            outputs=output_layer_names)
                    end.record()
                    print("Finish!")
                    print("Wait other process for synchronization...")
                    torch.cuda.synchronize()
                    
                    time_sec = round(start.elapsed_time(end) * (1e-3), 3)

                    tmp_time_per_img.append(time_sec)
                    tmp_pose_result_per_img.append(pose_result_per_img) # 이미지별 포즈 결과 append
                    print(f"--> pose inference time: {time_sec}\n")

                    # print(start.elapsed_time(end))

            # else:
            #     for idx, img in enumerate(img_data):
            #         with Timer(
            #         print_msg='Object detector inference time',
            #         is_batch=is_batch,
            #         model_type=pose_model_type,
            #         det_task=task,
            #         data_len=len(img_data)) as t:
            #             t.start()
            #             mmdet_results = inference_detector(det_model, img)
            #             t.finish()
            #             # det_time_results[task] = t.get_time()

            #             tmp_time.append(t.get_time())

            else:
                for idx, human_box in enumerate(human_box_results): #이미지별 박스 dictionary
                    # print("*"*100)
                    # # print(human_box_dict)
                    # exit()
                    with Timer(
                        print_msg=f"Pose Estimation Time Checking on {task.upper()} Human Detection Results... ",
                        is_batch=is_batch,
                        model_type=pose_model_type,
                        is_print_full_time=False,
                        det_task=task) as t:
                        t.start()
                        pose_result_per_img, heatmap_per_img = inference_top_down_pose_model(
                                pose_model,
                                img_data[idx],
                                human_box,
                                bbox_thr=args().bbox_thr,
                                format='xyxy',
                                dataset=datasets,
                                return_heatmap=return_heatmap,
                                outputs=output_layer_names)

                        t.finish()
                        times = t.get_time()
                        tmp_time_per_img.append(times)
                        tmp_pose_result_per_img.append(pose_result_per_img) # 이미지별 포즈 결과 append
                        print("--> pose inference time:", t.get_time())

            dict_pose_time_results[task] = tmp_time_per_img
            dict_pose_results[task] = tmp_pose_result_per_img

    elif pose_model_type == 'BottomUp':
        print(TextColor.YELLOW + "create " + TextColor.MAGENTA + "BottomUp pose model.... " + TextColor.RESET, end="  ")
        pose_model = init_pose_model(
            abs_pose_cfg, abs_pose_ckpt, device=args().device.lower()) # build the model and load checkpoint
        print(ShadeColor.BRIGHT_BLUE + "complete!" + TextColor.RESET  + "\n") 

        print("Start Bottom-up inferencing\n")
        tmp_time_per_img = []
        tmp_pose_result_per_img = []
        for idx, img in enumerate(img_data):
            if args().time_sync:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                print(f"Pose Estimation Time Checking on BottomUp Human Detection Results... ", end="")
                start.record()
                pose_result_per_img, heatmap_per_img = inference_bottom_up_pose_model(
                                pose_model,
                                img,
                                return_heatmap=return_heatmap,
                                outputs=output_layer_names)
                end.record()
                print("Finish!")
                print("Wait other process for synchronization...")
                torch.cuda.synchronize()
                
                time_sec = round(start.elapsed_time(end) * (1e-3), 3)

                tmp_time_per_img.append(time_sec)
                tmp_pose_result_per_img.append(pose_result_per_img) # 이미지별 포즈 결과 append
                print(f"--> pose inference time: {time_sec}\n")

            else:
                with Timer(
                        print_msg=f"Pose Estimation Time Checking using {args().model.upper()} ... ",
                        is_batch=is_batch,
                        model_type=pose_model_type,
                        is_print_full_time=False) as t:
                        t.start()
                        pose_result_per_img, heatmap_per_img = inference_bottom_up_pose_model(
                                pose_model,
                                img,
                                return_heatmap=return_heatmap,
                                outputs=output_layer_names)
                        t.finish()
                        times = t.get_time()
                        tmp_time_per_img.append(times)
                        tmp_pose_result_per_img.append(pose_result_per_img) # 이미지별 포즈 결과 append
                        print("--> pose inference time:", t.get_time())

        dict_pose_time_results[args().model.upper()] = tmp_time_per_img
        dict_pose_results[args().model.upper()] = tmp_pose_result_per_img
        
    print(ShadeColor.BRIGHT_BLUE + "complete!" + TextColor.RESET + "\n")

    print(dict_pose_time_results)
    pose_time_results = cal_total_mean_time(dict_pose_time_results)
    print(pose_time_results)
    print("*"*50)
    
    if pose_model_type == 'TopDown':
        print("detectin + estimation time: ",\
            round(det_total_results[args().det]['total_time'] + pose_time_results[args().det]['total_time'], 3))

    else:
        print("Bottomup pose estimation time: ",\
            round(pose_time_results[args().model.upper()]['total_time'], 3))


    print(ShadeColor.BRIGHT_BLUE + "complete!" + ShadeColor.RESET + "\n")


    pose_base_dir = f"/root/volume/{args().infer_data}/infer_results/{args().model}/{cfgfile_name}/"

    print(TextColor.YELLOW + "Visualize the inference map into the input image data" + TextColor.RESET)

    thickness = args().radius//2

    for task, pose_results in dict_pose_results.items():
        pose_save_dir = osp.join(pose_base_dir, f'{task}/results/{cases}/')
        mmcv.mkdir_or_exist(osp.abspath(pose_save_dir))
        for idx, box_kpt_data in enumerate(pose_results):
            out_file_name = f'vis_{task}_{img_name_lists[idx]}.png'
            out_file = osp.join(pose_save_dir, out_file_name)
            
            vis_pose_result(
                pose_model, # model
                img_data[idx], # image
                box_kpt_data, # pose results 
                radius = args().radius,
                thickness = thickness, # thickness of keypoints connection lines
                kpt_score_thr=args().kpt_thr,
                dataset=datasets,
                show=args().show,
                out_file=out_file)

    print(ShadeColor.BRIGHT_BLUE + "complete!" + ShadeColor.RESET)
    print(TextColor.GREEN + f"Save Path: {pose_base_dir}" + TextColor.RESET + "\n")

    print("-"*50, "\n")
    
    exit()


    print(TextColor.YELLOW + "Save experiment configuration information to yaml file... " + TextColor.RESET)
    exp_info_dict = {
        cases: 
            {
                'main_cfg': {
                    'pose_cfg': pose_cfg,
                    'pose_ckpt': pose_ckpt,
                    'det_cfg': args().det_cfg,
                    'det_ckpt': args().det_ckpt
                },

                'result_cfg': {
                    'volume_data': args().dataset,
                    'tovolume': args().tovolume,
                    'bbox-thr': args().bbox_thr,
                    'kpt-thr': args().kpt_thr,
                    'radius': args().radius,
                    'gpu_number': args().device
                },
                'save_info': {
                    'base_path': work_dir,
                    'image_list': out_file_list,
                } 
            }
    }

    yaml_file = "inference_info.yaml"
    save_file = osp.join(base_dir, yaml_file)
    # write_exp_info(save_file, exp_info_dict)
    dump_yaml(save_file, exp_info_dict)

    print(ShadeColor.BRIGHT_BLUE + "complete!" + ShadeColor.RESET + "\n")

    print(TextColor.BLACK + ShadeColor.WHITE + "Finish inference program" + TextColor.RESET + "\n\n")


if __name__ == '__main__':
    main()
