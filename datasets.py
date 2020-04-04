import os
import json
import random
import numpy as np
from collections import defaultdict, OrderedDict

import torch
import torchvision.transforms.functional as tvf

import utils.utils as Utils
import utils.augmentation as augUtils


class Dataset4ObjDet(torch.utils.data.Dataset):
    """
    Dataset for training object detection CNNs.

    Args:
        img_dir: str, imgs folder, e.g. 'someDir/COCO/train2017/'
        json_path: str, e.g. 'someDir/COCO/instances_train2017.json'
        bb_format: str, default: 'x1y1wh'
        img_size: int, target image size input to the YOLO, default: 608
        augmentation: bool, default: True
        only_person: bool, if true, non-person BBs are discarded. default: True
        debug: bool, if True, only one data id is selected from the dataset
    """
    def __init__(self, img_dir, json_path, bb_format, img_size, input_format,
                 augmentation=True, debug_mode=False):
        self.img_dir = img_dir
        self.img_size = img_size
        self.input_format = input_format
        self.enable_aug = augmentation
        # self.only_person = only_person
        # if only_person:
        #     print('Only train on person images and objects')
        self.skip_crowd = False

        self.img_ids = []
        self.imgid2info = dict()
        self.imgid2anns = defaultdict(list)
        self.catid2idx = []
        self.load_json(json_path, bb_format)

        if debug_mode:
            # self.img_ids = self.img_ids[0:1]
            self.img_ids = [222639]
            print(f"debug mode..., only train on one image: {self.img_ids[0]}")

    def load_json(self, json_path, bb_format):
        '''
        laod json file to self.img_ids, self.imgid2anns
        '''
        print(f'loading annotations {json_path} into memory...')
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        catgys = json_data['categories']
        self.catid2idx = dict([(cat['id'],idx) for idx,cat in enumerate(catgys)])

        for ann in json_data['annotations']:
            # get width and height
            if bb_format == 'x1y1wh':
                assert len(ann['bbox']) == 4
                # The dataset is using (x1,y1,w,h). Converting to (cx,cy,w,h)
                ann['bbox'][0] = ann['bbox'][0] + ann['bbox'][2] / 2
                ann['bbox'][1] = ann['bbox'][1] + ann['bbox'][3] / 2
            elif bb_format == 'cxcywh':
                assert len(ann['bbox']) == 4
            else:
                raise Exception('Bounding box format not supported')
            cat_id = self.catid2idx[ann['category_id']]
            ann['gt'] = torch.Tensor([cat_id] + ann['bbox'])
            self.imgid2anns[ann['image_id']].append(ann)

        for img in json_data['images']:
            img_id = img['id']
            anns = self.imgid2anns[img_id]
            if self.skip_crowd:
                # if there is crowd gt, skip this image
                if any(ann['iscrowd'] for ann in anns):
                    continue
            # if only for person detection
            # if self.only_person:
            #     # select the images which contain at least one person
            #     if not any(ann['category_id']==1 for ann in anns):
            #         continue
            #     # and ignore all other categories
            #     self.imgid2anns[img_id] = [a for a in anns if a['category_id']==1]
            # otherwise, keep all 80 classes
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
        img = Utils.imread_pil(img_path)
        ori_w, ori_h = img.width, img.height

        labels = []
        for _, ann in enumerate(self.imgid2anns[img_id]):
            # if self.only_person and ann['category_id'] != 1:
            #     continue
            labels.append(ann['gt'])
        # if self.only_person:
        #     assert (labels[:,0] == 0).all()
        labels = torch.stack(labels, dim=0) if labels else torch.zeros(0,5)
        # each row of labels is [category, x, y, w, h]
        # augmentation
        if self.enable_aug:
            img, labels = self.augment_PIL(img, labels)
        # pad to square
        img, labels, pad_info = Utils.rect_to_square(img, labels, self.img_size,
                                        pad_value=0, aug=self.enable_aug)
        # Remove annotations which are too small
        label_areas = labels[:,3] * labels[:,4]
        labels = labels[label_areas > 64]
        # Convert PIL.image into torch.tensor with shape (3,h,w)
        img = tvf.to_tensor(img)
        # Noise augmentation
        if self.enable_aug:
            # blur = [augUtils.random_avg_filter, augUtils.max_filter,
            #         augUtils.random_gaussian_filter]
            # if np.random.rand() > 0.7:
            #     blur_func = random.choice(blur)
            #     img = blur_func(img)
            # if np.random.rand() > 0.6:
                # img = augUtils.add_gaussian(img, max_var=0.002)
            if np.random.rand() > 0.6:
                img = augUtils.add_saltpepper(img, max_p=0.02)
        # Convert into desired input format, e.g., normalized
        img = Utils.format_tensor_img(img, code=self.input_format)
        # Debugging
        if (labels[:,1:5] >= self.img_size).any():
            print('Warning: some x,y in ground truth are greater than image size')
            print('image path:', img_path)
        if (labels[:,1:5] < 0).any():
            print('Warning: some x,y in ground truth are smaller than 0')
            print('image path:', img_path)
        labels[:,1:5].clamp_(min=0)
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