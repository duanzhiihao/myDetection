# API for the object detectors
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import randint

import torch
import torchvision.transforms.functional as tvf

from models.registry import name_to_model
import utils.utils as Utils
import utils.visualization as visUtils


class Detector():
    def __init__(self, model_name=None, weights_path=None, model=None, **kwargs):
        assert torch.cuda.is_available()

        self.conf_thres = kwargs.get('conf_thres', 0.5)
        self.nms_thres = kwargs.get('nms_thres', 0.45)

        if model:
            self.model = model
            return
        else:
            model = name_to_model(model_name)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of parameters:', total_params)
        if weights_path:
            model.load_state_dict(torch.load(weights_path)['model'])
        self.model = model.cuda()
        self.model.eval()
        
    def predict_imgDir(self, img_dir, **kwargs):
        '''
        Run the object detector on a sequence of images
        
        Args:
            img_dir: str
            See _predict_pil() for optinal arguments
        '''
        self.model.eval()
        img_names = os.listdir(img_dir)
        detection_json = []
        for imname in tqdm(img_names):
            impath = os.path.join(img_dir, imname)
            detections = self.detect_one(img_path=impath, **kwargs)

            assert imname[-4] == '.'
            img_id = int(imname[:-4]) if imname[:-4].isdigit() else imname[:-4]
            detection_json += detections.to_coco_json(img_id=img_id)

        return detection_json

    def detect_one(self, **kwargs): # img_path, test_aug=None, input_size=1024):
        '''
        object detection in one single image. Predict and show the results

        Args:
            (img_path: str) or (pil_img: PIL.Image)
            return_img (default: False): bool
            show_img (default: False): bool
            See _predict_pil() for more optinal arguments
        '''
        if 'img_path' in kwargs:
            img = Utils.imread_pil(kwargs['img_path'])
        else:
            assert 'pil_img' in kwargs
            img = kwargs.pop('pil_img')

        detections = self._predict_pil(img, **kwargs)

        if kwargs.get('return_img', False):
            np_img = np.array(img)
            detections.draw_on_np(np_img, class_map='COCO', print_dt=False)
            return np_img
        if kwargs.get('show_img', False):
            np_img = np.array(img)
            detections.draw_on_np(np_img, class_map='COCO', print_dt=True)
            plt.imshow(np_img)
            plt.show()
        return detections

    def _predict_pil(self, pil_img, **kwargs):
        '''
        Args:
            test_aug: str, test-time augmentation, can be: 'h'
            input_size: int, default: 640
            conf_thres: float, confidence threshold
        '''
        assert isinstance(pil_img, Image.Image), 'input must be a PIL.Image'
        # test_aug = kwargs['test_aug'] if 'test_aug' in kwargs else None
        input_size = kwargs.get('input_size', None)
        to_square = kwargs.get('to_square', False)
        conf_thres = kwargs.get('conf_thres', self.conf_thres)
        nms_thres = kwargs.get('nms_thres', self.nms_thres)

        # resize such that the shorter side = input_size
        ori_shorter = min(pil_img.height, pil_img.width)
        if to_square:
            assert input_size > 0
            pil_img, _, pad_info = Utils.rect_to_square(pil_img, None, input_size)
        elif input_size:
            pil_img = tvf.resize(pil_img, input_size)
        # convert to tensor
        t_img = tvf.to_tensor(pil_img)
        t_img = Utils.format_tensor_img(t_img, code=self.model.input_format)

        input_ = t_img.unsqueeze(0)
        assert input_.dim() == 4
        with torch.no_grad():
            dts = self.model(input_.cuda())
        assert isinstance(dts, list)
        dts = dts[0]
        # post-processing
        dts.cpu_()
        dts = dts[dts.scores >= conf_thres]
        if len(dts) > 1000:
            _, idx = torch.topk(dts.scores, k=1000)
            dts = dts[idx]
        dts = dts.nms(nms_thres=nms_thres)
        # np_img = np.array(tvf.to_pil_image(input_.squeeze()))
        # visualization.draw_cocobb_on_np(np_img, dts, print_dt=True)
        # plt.imshow(np_img)
        # plt.show()
        if to_square:
            dts.bboxes = Utils.detection2original(dts.bboxes, pad_info.squeeze())
        elif input_size:
            dts.bboxes = dts.bboxes / input_size * ori_shorter
        return dts
