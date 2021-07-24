import os
import yaml
import argparse
import os.path as osp

import mmcv
import torch

from cskim_custom.utils.exception import *

class TextColor:
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m' 
    BRIGHT_GREEN = '\033[92m' 
    BRIGHT_YELLOW = '\033[93m' 
    BRIGHT_BLUE = '\033[94m' 
    BRIGHT_MAGENTA = '\033[95m' 
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m' 

    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'

    BOLD = '\033[1m'
    RESET = '\033[0m'


class ShadeColor:
    BRIGHT_BLACK = '\033[100m'
    BRIGHT_RED = '\033[101m' 
    BRIGHT_GREEN = '\033[102m' 
    BRIGHT_YELLOW = '\033[103m' 
    BRIGHT_BLUE = '\033[104m' 
    BRIGHT_MAGENTA = '\033[105m' 
    BRIGHT_CYAN = '\033[106m'
    BRIGHT_WHITE = '\033[107m' 

    BLACK = '\033[40m'
    RED = '\033[41m'
    GREEN = '\033[42m'
    YELLOW = '\033[43m'
    BLUE = '\033[44m'
    MAGENTA = '\033[45m'
    CYAN = '\033[46m'
    WHITE = '\033[47m'
    UNDERLINE = '\033[4m'

    RESET = '\033[0m'


def check_case_len(case):
    if len(case) > 1:
        return case

    elif len(case) == 1:
        res = case.zfill(2)
    
    return res


def _get_single_image(img_data):
    single_img = os.path.abspath(img_data)
    return [single_img]


def _get_multi_images(img_data):
    return multi_imgs


def get_image_list(img_data):
    img_fmt = ['.png', '.jpg', '.jpeg']
    file_fmt = osp.splitext(img_data)[1]

    if osp.isfile(img_data):
        if file_fmt not in img_fmt: # if img_data is not a image, is a text file(txt, json, etc.)
            img_list = mmcv.load(img_data)
            return img_list
        else:
            img_path = os.path.abspath(img_data)
            return [img_path]

    elif osp.isdir(img_data):
        abs_dir_path = osp.abspath(img_data)
        
        img_list = [osp.join(abs_dir_path, img) for img in os.listdir(abs_dir_path) if osp.splitext(img)[1] in img_fmt]  
        img_list = sorted(img_list)
        
        return img_list


def is_file_exist(file_path):
    return osp.isfile(file_path)


def load_yaml(yaml_path):
    data = mmcv.load(yaml_path, file_format='yaml')
    return data


def dump_yaml(yaml_file, data, file_format='yaml'):
    if not is_file_exist(yaml_file) or os.stat(yaml_file).st_size==0:
        mmcv.dump(data, yaml_file, default_flow_style=False, sort_keys=False)

    # append to yaml file
    else:
        with open(yaml_file, 'r') as f:
            cur_yaml = yaml.safe_load(f)
            cur_yaml.update(data)

        with open(yaml_file, 'w') as f:
            yaml.safe_dump(cur_yaml, f, sort_keys=False)


def get_base_pose_info(cfg_path, model, dataset, cfgnum):
    data = load_yaml(cfg_path)
    cfg, ckpt = data['Models'][model]['dataset'][dataset]['configs'][cfgnum]
    
    return cfg, ckpt

def get_det_info(det_cfg_path, detector, det_num=1):
    data = load_yaml(det_cfg_path)
            
    cfg, ckpt = data['Models'][detector.lower()]['dataset']['coco']['configs'][det_num]

    return cfg, ckpt


def get_base_all_det_info(cfg_path, detector, det_num=1):
    data = load_yaml(cfg_path)
    task_dict = dict()

    for task in list(data['Models'].keys()):
        # cfg, ckpt = data['Models'][task]['dataset']['coco']['configs'][1]
        task_dict[task.lower()] = data['Models'][task]['dataset']['coco']['configs'][det_num]

    return task_dict


def make_base_path_for_pose(model, dataset, pose_cfg, pose_ckpt):
    root = f'/root/mmpose/cskim_custom/models'
    cfg_base_path = f'{root}/{model}/{dataset}/configs/{pose_cfg}'
    ckpt_base_path = f'{root}/{model}/{dataset}/ckpt/{pose_ckpt}'

    return cfg_base_path, ckpt_base_path


def set_specific_gpus(device):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu = device.lower() == 'cpu'
    
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability


def calculate_time(
    correct_data_len,
    det_time_results,
    pose_time_results):
    det_total_time = sum(det_time_results)
    det_mean_time = round((det_total_time/correct_data_len)*(1e-3), 4)
    det_total_time = round(det_total_time*(1e-3), 4)

    pose_total_time = sum(pose_time_results)
    pose_mean_time = round((pose_total_time/correct_data_len)*(1e-3), 4)
    pose_total_time = round(pose_total_time*(1e-3), 4)

    all_time_results = []
    for each_time in zip(det_time_results, pose_time_results):
        all_time_results.append(sum(each_time))
    
    all_total_time = sum(all_time_results)
    all_mean_time = round((all_total_time/correct_data_len)*(1e-3), 4)
    all_total_time = round(all_total_time*(1e-3), 4)
    
    

    return det_total_time, det_mean_time, pose_total_time, pose_mean_time, all_total_time, all_mean_time

def save_error_info(save_path, img_path, img_name):
    error_save_path = osp.join(save_path, f'{img_name}.png')
    img = mmcv.imread(img_path)
    mmcv.imwrite(img, error_save_path)
    


def is_cropping(resize_scale, h_crop, w_crop):
    if resize_scale == 'origin':
        assert (h_crop == 1 and w_crop == 1), (
            f'the resize scale is origin, but both H and W rate are not 1'
        )
        is_crop = False

    elif resize_scale == 'cropped':
        assert (h_crop > 1 or w_crop > 1), (
            f'the resize scale is cropped, but both H and W rate are 1'
        )
        is_crop = True

    # if not integer, it will raise "ValueError" because of not correct input
    elif isinstance(int(resize_scale), int):
        assert int(resize_scale) > 32, (
            f'the specific resize scale value is too small (value: {resize_scale})')
        assert int(resize_scale) > 1, (
            f'the specific resize scale value is negative value (value: {resize_scale})')
        is_crop = True

    else:
        raise CropSettingError(
                f"the current value of argument [resize_scale] is {resize_scale}")

    return is_crop