import os
import json
import random
import numpy as np
from collections import defaultdict, OrderedDict

import torch
import torchvision.transforms.functional as tvf

import utils.image_ops as imgUtils
import utils.augmentation as augUtils
from utils.structures import ImageObjects


class Dataset4ObjDet(torch.utils.data.Dataset):
    """
    Dataset for training object detection CNNs.

    Args:
        img_dir: str, imgs folder, e.g. 'someDir/COCO/train2017/'
        json_path: str, e.g. 'someDir/COCO/instances_train2017.json'
        bb_format: str, default: 'x1y1wh'
        img_size: int, target image size input to the YOLO, default: 608
        augmentation: bool, default: True
        debug: bool, if True, only one data id is selected from the dataset
    """
    # def __init__(self, img_dir, json_path, bb_format, img_size, input_format,
    #              augmentation=True):
    def __init__(self, cfg: dict):
        self.img_dir = cfg['img_dir']
        self.img_size = cfg['img_size']
        self.input_format = cfg['input_image_format']
        self.enable_aug = cfg['enable_aug']
        self.skip_crowd = False
        self.config = cfg

        self.img_ids = []
        self.imgid2info = dict()
        self.imgid2anns = defaultdict(list)
        self.catid2idx = []
        self.load_json(cfg['json_path'], cfg['ann_bbox_format'])

    def load_json(self, json_path, ann_bbox_format):
        '''
        laod json file to self.img_ids, self.imgid2anns
        '''
        print(f'loading annotations {json_path} into memory...')
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        catgys = json_data['categories']
        self.catid2idx = dict([(cat['id'],idx) for idx,cat in enumerate(catgys)])

        for ann in json_data['annotations']:
            # Parse bounding box annotation
            if ann_bbox_format == 'x1y1wh':
                assert len(ann['bbox']) == 4
                # The dataset is using (x1,y1,w,h). Converting to (cx,cy,w,h)
                ann['bbox'][0] = ann['bbox'][0] + ann['bbox'][2] / 2
                ann['bbox'][1] = ann['bbox'][1] + ann['bbox'][3] / 2
                self.bb_format = 'cxcywh'
            elif ann_bbox_format == 'cxcywh':
                assert len(ann['bbox']) == 4
                self.bb_format = 'cxcywh'
            elif ann_bbox_format == 'cxcywhd':
                assert len(ann['bbox']) == 5
                self.bb_format = 'cxcywhd'
            else: raise Exception('Bounding box format is not supported')
            cat_idx = self.catid2idx[ann['category_id']]
            ann['_gt'] = torch.Tensor([cat_idx] + ann['bbox'])
            self.imgid2anns[ann['image_id']].append(ann)

        for img in json_data['images']:
            img_id = img['id']
            anns = self.imgid2anns[img_id]
            if self.skip_crowd:
                # if there is crowd gt, skip this image
                if any(ann['iscrowd'] for ann in anns):
                    continue
            self.img_ids.append(img_id)
            self.imgid2info[img['id']] = img
        debug = 1

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index
        """
        # laod the image
        img_id = self.img_ids[index]
        img_name = self.imgid2info[img_id]['file_name']
        img_path = os.path.join(self.img_dir, img_name)
        img = imgUtils.imread_pil(img_path)
        ori_w, ori_h = img.width, img.height

        labels = []
        for _, ann in enumerate(self.imgid2anns[img_id]):
            labels.append(ann['_gt'])
        labels = torch.stack(labels, dim=0) if labels else torch.zeros(0,6)
        # each row of labels is [category, cx, cy, w, h, (degree)]
        # augmentation
        if self.enable_aug:
            img, labels = self.augment_PIL(img, labels)
        # pad to square
        img, labels, pad_info = imgUtils.rect_to_square(img, labels, self.img_size,
                                        pad_value=0, aug=self.enable_aug)
        # Remove annotations which are too small
        label_areas = labels[:,3] * labels[:,4]
        labels = labels[label_areas >= 50]
        # Convert PIL.image into torch.tensor with shape (3,h,w)
        img = tvf.to_tensor(img)
        # Noise augmentation
        if self.enable_aug:
            # blur = [augUtils.random_avg_filter, augUtils.max_filter,
            #         augUtils.random_gaussian_filter]
            # if np.random.rand() > 0.7:
            #     blur_func = random.choice(blur)
            #     img = blur_func(img)
            if np.random.rand() > 0.6:
                img = augUtils.add_saltpepper(img, max_p=0.02)
        # Convert into desired input format, e.g., normalized
        img = imgUtils.format_tensor_img(img, code=self.input_format)
        # Debugging
        if (labels[:,1:5] >= self.img_size).any():
            print('Warning: some x,y in ground truth are greater than image size')
            print('image path:', img_path)
        if (labels[:,1:5] < 0).any():
            print('Warning: some x,y in ground truth are smaller than 0')
            print('image path:', img_path)
        labels[:,1:5].clamp_(min=0)
        labels = ImageObjects(bboxes=labels[:,1:], cats=labels[:,0].long(),
                              bb_format=self.bb_format, img_size=img.shape[1:3])
        assert img.dim() == 3 and img.shape[0] == 3 and img.shape[1] == img.shape[2]
        return img, labels, img_id, pad_info

    def augment_PIL(self, img, labels):
        if np.random.rand() > 0.5:
            img = tvf.adjust_brightness(img, uniform(0.6, 1.4))
        if np.random.rand() > 0.5:
            img = tvf.adjust_contrast(img, uniform(0.5, 1.5))
        if np.random.rand() > 0.5:
            img = tvf.adjust_hue(img, uniform(-0.1, 0.1))
        if np.random.rand() > 0.5:
            factor = uniform(0,2)
            if factor > 1:
                factor = 1 + uniform(0, 2)
            img = tvf.adjust_saturation(img, factor) # 0 ~ 3
        # if np.random.rand() > 0.5:
        #     img = tvf.adjust_gamma(img, uniform(0.5, 3))
        # horizontal flip
        if np.random.rand() > 0.5:
            img, labels = augUtils.hflip(img, labels)
        if self.bb_format in {'cxcywhd'}:
            # vertical flip
            if np.random.rand() > 0.5:
                img, labels = augUtils.vflip(img, labels)
            # random rotation
            rand_deg = np.random.rand() * 360
            if self.config['rotation_expand']:
                img, labels = augUtils.rotate(img, rand_deg, labels, expand=True)
            else:
                img, labels = augUtils.rotate(img, rand_deg, labels, expand=False)
            return img, labels

        return img, labels

    @staticmethod
    def collate_func(batch):
        img_batch = torch.stack([items[0] for items in batch])
        label_batch = [items[1] for items in batch]
        ids_batch = [items[2] for items in batch]
        pad_info_batch = [items[3] for items in batch]
        return img_batch, label_batch, ids_batch, pad_info_batch


def uniform(a, b):
    return a + np.random.rand() * (b-a)