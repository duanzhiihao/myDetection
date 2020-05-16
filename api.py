# API for the object detectors
import os
import numpy as np
import cv2
import PIL.Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import randint

import torch
import torchvision.transforms.functional as tvf

from models.general import name_to_model
import utils.image_ops as imgUtils
import utils.visualization as visUtils


class Detector():
    def __init__(self, model_name:str=None, model_and_cfg:tuple=None,
                       weights_path:str=None, cpu=False):
        if model_and_cfg:
            self.model, cfg = model_and_cfg
            # self.model.eval()
        else:
            self.model, cfg = name_to_model(model_name)
            self.model.eval()
            if not cpu:
                self.model = self.model.cuda()

        self._init_preprocess(cfg)
        self._init_postprocess(cfg)
        
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Number of parameters:', n_params)
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path)['model'])
        
        self.on_cpu = cpu
    
    def _init_preprocess(self, cfg):
        preprocess_name = cfg['test.preprocessing']
        self.divisibe = cfg['general.input_divisibility']
        self.input_size = cfg.get('test.default_input_size', None)
        self.preprocess = preprocess_name
    
    def _init_postprocess(self, cfg):
        self.conf_thres = cfg['test.default_conf_thres']
        self.nms_thres = cfg['test.nms_thres']

    def evaluation_predict(self, eval_info, **kwargs):
        '''
        Args:
            eval_info: list of (img_path, img_id)
            See _predict_pil() for optinal arguments
        '''
        detection_json = []
        for (impath, imgId) in tqdm(eval_info):
            detections = self.detect_one(img_path=impath, **kwargs)
            detection_json += detections.to_json(img_id=imgId)
        return detection_json
        
    def predict_imgDir(self, img_dir, **kwargs):
        '''
        Run the object detector on a sequence of images
        
        Args:
            img_dir: str
            See _predict_pil() for optinal arguments
        '''
        img_names = os.listdir(img_dir)
        detection_json = []
        for imname in tqdm(img_names):
            impath = os.path.join(img_dir, imname)
            detections = self.detect_one(img_path=impath, **kwargs)

            assert imname[-4] == '.'
            img_id = int(imname[:-4]) if imname[:-4].isdigit() else imname[:-4]
            detection_json += detections.to_json(img_id=img_id)

        return detection_json

    def detect_one(self, **kwargs):
        '''
        object detection in one single image. Predict and show the results

        Args:
            (img_path: str) or (pil_img: PIL.Image)
            return_img (default: False): bool
            show_img (default: False): bool
            See _predict_pil() for more optinal arguments
        '''
        assert 'pil_img' in kwargs or 'img_path' in kwargs
        img = kwargs.pop('pil_img', None) or \
              imgUtils.imread_pil(kwargs.pop('img_path'))

        detections = self._predict_pil(img, **kwargs)

        if kwargs.get('return_img', False):
            np_img = np.array(img)
            detections.draw_on_np(np_img, class_map='COCO', print_dt=False)
            return np_img
        if kwargs.get('show_img', False):
            np_img = np.array(img)
            detections.draw_on_np(np_img, class_map='COCO', print_dt=True)
            plt.figure(figsize=(8,8))
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
        assert isinstance(pil_img, PIL.Image.Image), 'input must be a PIL.Image'
        # test_aug = kwargs['test_aug'] if 'test_aug' in kwargs else None
        pre_proc = kwargs.get('preprocessing', self.preprocess)
        input_size = kwargs.get('input_size', self.input_size)
        conf_thres = kwargs.get('conf_thres', self.conf_thres)
        nms_thres = kwargs.get('nms_thres', self.nms_thres)

        # pre-process the input image
        pil_img, pad_info = self._preprocess_pil(pil_img, pre_proc, input_size)
        # convert to tensor
        t_img = tvf.to_tensor(pil_img) # (H,W,3), 0-1 float
        t_img = imgUtils.format_tensor_img(t_img, code=self.model.input_format)

        input_ = t_img.unsqueeze(0)
        assert input_.dim() == 4
        if not self.on_cpu:
            input_ = input_.cuda()
        with torch.no_grad():
            dts = self.model(input_)
        assert isinstance(dts, list)
        dts = dts[0]
        # post-processing
        dts.cpu_()
        dts = dts[dts.scores >= conf_thres]
        if len(dts) > 1000:
            _, idx = torch.topk(dts.scores, k=1000)
            dts = dts[idx]
        # pil_img = imgUtils.tensor_img_to_pil(input_[0], self.model.input_format)
        # np_im = np.array(pil_img)
        # dts.draw_on_np(np_im, imshow=True)
        dts = dts.nms(nms_thres=nms_thres)
        if pad_info is not None:
            dts.bboxes_to_original_(pad_info)
        return dts

    def _preprocess_pil(self, pil_img, pre_proc_name,
                        input_size=None) -> PIL.Image.Image:
        assert isinstance(pil_img, PIL.Image.Image), 'input must be a PIL.Image'
        assert isinstance(self.divisibe, int)
        ori_h, ori_w = pil_img.height, pil_img.width
        
        if pre_proc_name == 'pad_divisible':
            # zero-padding at the right and bottom such that
            # both side of the image are divisible by `div`
            pil_img = imgUtils.pad_to_divisible(pil_img, self.divisibe)
            pad_info = None
        elif pre_proc_name == 'resize_pad_divisible':
            assert input_size is not None
            # resize the image such that the SHORTER side of the image
            # equals to desired input size; then pad it to be divisible
            pil_img = tvf.resize(pil_img, input_size)
            new_h, new_w = pil_img.height, pil_img.width
            pil_img = imgUtils.pad_to_divisible(pil_img, self.divisibe)
            assert min(new_h, new_w) == input_size
            pad_info = (ori_w, ori_h, 0, 0, new_w, new_h)
        elif pre_proc_name == 'resize_pad_square':
            assert input_size is not None
            # resize the image such that the LONGER side of the image
            # equals to desired input size; then pad it to be square
            pil_img, _, pad_info = imgUtils.rect_to_square(pil_img, None,
                                    input_size, pad_value=0, aug=False)
            pad_info = pad_info
        else:
            raise Exception('Unknown preprocessing name')
        return pil_img, pad_info
