import warnings

import mmcv
import numpy as np
import torch
import os
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


class DetectionLoadImage:
    """Deprecated.

    A simple pipeline to load image.
    """

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        warnings.simplefilter('once')
        warnings.warn('`LoadImage` is deprecated and will be removed in '
                      'future releases. You may use `LoadImageFromWebcam` '
                      'from `mmdet.datasets.pipelines.` instead.')
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        # convert string to one-element list
        imgs = [imgs]
        is_batch = False


    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # if single image, the type of imgs is string (file path)
    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)

        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)
    
    data = collate(datas, samples_per_gpu=len(imgs))

    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # forward the model
    with torch.no_grad():
        start.record()
        results = model(return_loss=False, rescale=True, **data)
        end.record()

    torch.cuda.synchronize()                   
    det_time_sec = start.elapsed_time(end)

    if not is_batch:
        return results[0], det_time_sec# type: list
    else:
        return results, det_time_sec # type: list


      
def _get_cropped_box(human_bboxes, h_rate, w_rate):
    full_box_results = []
    crop_box_results = []

    for bbox in human_bboxes:
        # full_box = {}
        # full_box['bbox'] = bbox
        # full_box_results.append(full_box)

        bbox_score = bbox[4]
        box_crop_h = (bbox[3] - bbox[1]) / h_rate
        box_crop_w = (bbox[2] - bbox[0]) / w_rate

        # ex) h_rate: 4 / w_rate: 4 ==> 16 points
        std_topleft_x = bbox[0]
        std_topleft_y = bbox[1]
        for h in range(h_rate):
            tmp_x = std_topleft_x
            tmp_y = std_topleft_y

            for w in range(w_rate):
                crop_box = {}

                bottomright_x = tmp_x + (box_crop_w)
                bottomright_y = tmp_y + (box_crop_h)

                crop_box['bbox'] = [
                    tmp_x, tmp_y, 
                    bottomright_x, bottomright_y, 
                    bbox_score, [int(box_crop_h), int(box_crop_w)]
                    ]
                crop_box_results.append(crop_box)

                # the value of tmp_y is fixed
                tmp_x += box_crop_w

            # the value of std_topleft_x is fixed
            std_topleft_y += box_crop_h
            

    return crop_box_results


def _get_bboxes_in_multi_images(det_results, cat_id):
    bboxes_per_image = []
    for det_img in det_results:
        bboxes_per_image.append(det_img[cat_id - 1])

    return bboxes_per_image


def _get_bboxes_in_single_image(human_bboxes):
    person_results = []
    
    for bbox in human_bboxes:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results


def process_mmdet_results(
    mmdet_results, 
    is_crop, 
    h_rate,
    w_rate,
    cat_id=1):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 1 for human)
    :return: a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        # print("tuple")
        det_results = mmdet_results[0]
    else:
        # print("no tuple")
        det_results = mmdet_results

    human_bboxes = det_results[cat_id - 1]

    if is_crop:
        crop_box_results = _get_cropped_box(
            human_bboxes, h_rate, w_rate)
        
        return crop_box_results
        
    else:
        person_results = _get_bboxes_in_single_image(human_bboxes)
        
        return person_results


async def async_inference_detector(model, imgs):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray): Either image files or loaded images.

    Returns:
        Awaitable detection results.
    """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    results = await model.aforward_test(rescale=True, **data)
    return results


# /root/mmdetection/mmdet/models/detectors/base.py
def show_result(img,
                result,
                score_thr=0.3,
                bbox_color=(72, 101, 241), # order: B/G/R
                text_color=(255, 255, 255),
                mask_color=None,
                thickness=2,
                font_size=10,
                class_names = get_classes('coco'),
                only_person=True,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """

        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None

        if only_person:
            bboxes = bbox_result[0]
            labels = np.full(len(bboxes), 0, dtype=np.int32)
            class_names = ['Person']

        else:
            # (80,)의 값 중 빈 배열을 제거하고 값이 존재하는 인덱스의 값을 추출하여 그 인덱스들 만으로 새로운 배열 생성. 열 크기는 동일
            bboxes = np.vstack(bbox_result) 
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32) # 예측된 인덱스 위치마다 해당 인덱스의 값 부여. 즉 클래스 번호 부여
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels) # 클래스 부여 받은 값만 추출하여 새로운 1차원 배열 생성
        
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=class_names,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img


def run_detection(detector, img):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    mmdet_results, det_infer_time = inference_detector(detector, img)

    return mmdet_results, det_infer_time


