import os
import json
import random
from collections import defaultdict
import PIL.Image
import torch
import torchvision.transforms.functional as tvf

from .base import InfiniteDataset
import utils.image_ops as imgUtils
import utils.augmentation as augUtils
import utils.mask_ops as maskUtils
from utils.structures import ImageObjects


class ImageDataset(InfiniteDataset):
    """
    Dataset for training object detection CNNs.

    Args:
        dataset_cfg:
            img_dir: str, imgs folder
            ann_path: str, path to the annotation file
            ann_bbox_format: str, e.g., 'x1y1wh' for COCO
        global_cfg: global config
    """
    def __init__(self, dataset_cfg: dict, global_cfg: dict):
        self.img_dir            = dataset_cfg['img_dir']
        self.img_size           = global_cfg['train.initial_imgsize']
        self.input_format       = global_cfg['general.input_format']
        self.aug_setting        = global_cfg['train.data_augmentation']
        self.input_divisibility = global_cfg['general.input_divisibility']

        self.skip_crowd_ann = True
        self.skip_crowd_img = False
        self.skip_empty_img = True

        self.HEM = global_cfg['train.hard_example_mining']

        self.img_ids    = []
        self.imgId2info = dict()
        self.imgId2anns = defaultdict(list)
        self.catId2idx  = dict()
        self._load_json(dataset_cfg['ann_path'], dataset_cfg['ann_bbox_format'])

    def _load_json(self, json_path, ann_bbox_format):
        '''load json file'''
        print(f'Loading annotations {json_path} into memory...')
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        if ann_bbox_format in {'x1y1wh', 'cxcywh'}:
            self.bb_format = 'cxcywh'
            self.bb_param = 4
        elif ann_bbox_format == 'cxcywhd':
            self.bb_format = 'cxcywhd'
            self.bb_param = 5
        else:
            raise Exception('Bounding box format is not supported')

        for img in json_data['images']:
            self.imgId2info[img['id']] = img

        for idx, cat in enumerate(json_data['categories']):
            self.catId2idx[cat['id']] = idx
            
        for ann in json_data['annotations']:
            # Parse bounding box annotation
            assert len(ann['bbox']) == self.bb_param
            if self.skip_crowd_ann and ann['iscrowd']:
                continue
            # If the dataset is using (x1,y1,w,h), convert to (cx,cy,w,h)
            if ann_bbox_format == 'x1y1wh':
                ann['bbox'][0] = ann['bbox'][0] + ann['bbox'][2] / 2
                ann['bbox'][1] = ann['bbox'][1] + ann['bbox'][3] / 2
            # category inddex
            ann['cat_idx'] = self.catId2idx[ann['category_id']]
            # segmentation mask
            imgInfo = self.imgId2info[ann['image_id']]
            imh, imw = imgInfo['height'], imgInfo['width']
            if ann['segmentation'] != []:
                ann['rle'] = maskUtils.segm2rle(ann.pop('segmentation'), imh, imw)
            self.imgId2anns[ann['image_id']].append(ann)

        for img in json_data['images']:
            img_id = img['id']
            anns = self.imgId2anns[img_id]
            if self.skip_crowd_img and any(ann['iscrowd'] for ann in anns):
                # if there is crowd gt, skip this image
                continue
            if self.skip_empty_img and len(anns) == 0:
                # if there is no object in this image, skip this image
                continue
            self.img_ids.append(img_id)

        self._length = len(self.img_ids)
        if self.HEM == 'epoch_shuffle':
            self.hem_state = {
                'iter': -1,
                'order': torch.randperm(self._length)
            }
        elif self.HEM == 'hardest':
            raise NotImplementedError()
            self.hem_state = {
                'iter': -1,
                'APs': torch.ones(self._length)
            }
        elif self.HEM == 'probability':
            raise NotImplementedError()
            self.hem_state = {
                'iter': -1,
                'APs': torch.ones(self._length)
            }
        else:
            raise NotImplementedError()
        # breakpoint()

    def get_hem_index(self):
        assert self.hem_state is not None
        self.hem_state['iter'] += 1

        if self.HEM == 'epoch_shuffle':
            _iter = self.hem_state['iter']
            if _iter >= self._length:
                _iter = 0
                self.hem_state['order'] = torch.randperm(self._length)
            index = self.hem_state['order'][_iter].item()
        
        return index

    def __getitem__(self, _):
        """
        Get an image-label pair
        """
        mosaic = self.aug_setting['mosaic'] if self.aug_setting is not None else None
        if mosaic:
            raise NotImplementedError()
            index = random.randint(0, len(self.img_ids)-1)
            pairs = []
            for _ in range(4):
                img_label_pair = self._load_single_pil(index, to_square=False)
                pairs.append(img_label_pair)
            img_label_pair = augUtils.mosaic(pairs, self.img_size)
        else:
            index = self.get_hem_index()
            img_label_pair = self._load_single_pil(index, to_square=True)
        
        img, labels, img_id, pad_info = img_label_pair
        # Convert PIL.image to torch.Tensor with shape (3,h,w) if it's not
        if isinstance(img, PIL.Image.Image):
            img = tvf.to_tensor(img)
        else:
            assert isinstance(img, torch.FloatTensor)
        assert isinstance(labels, ImageObjects)
        # Noise augmentation
        if self.aug_setting is not None:
            # blur = [augUtils.random_avg_filter, augUtils.max_filter,
            #         augUtils.random_gaussian_filter]
            # if torch.rand(1).item() > 0.7:
            #     blur_func = random.choice(blur)
            #     img = blur_func(img)
            if torch.rand(1).item() > 0.7:
                p = self.aug_setting.get('satpepper_noise_density', 0.02)
                img = augUtils.add_saltpepper(img, max_p=p)
        # Convert into desired input format, e.g., normalized
        img = imgUtils.format_tensor_img(img, code=self.input_format)
        # Remove annotations which are too small
        label_areas = labels.bboxes[:,2] * labels.bboxes[:,3]
        labels = labels[label_areas >= 32]

        # sanity check before return
        if (labels.bboxes[:,0:2] > self.img_size).any():
            print('Warning: some x,y in ground truth are greater than image size')
            print('image id:', img_id)
        # if (labels.bboxes[:,2:4] > self.img_size).any():
        #     print('Warning: some w,h in ground truth are greater than image size')
        #     print('image path:', img_path)
        if (labels.bboxes[:,0:4] < 0).any():
            print('Warning: some bbox in ground truth are smaller than 0')
            print('image id:', img_id)
        labels.bboxes[:,0:4].clamp_(min=0)
        assert img.dim() == 3 and img.shape[1] == img.shape[2]
        return img, labels, img_id, pad_info

    # def _load_concat_frames(self, index, to_square=True) -> tuple:
    #     raise NotImplementedError()
    #     # load the image
    #     img_id = self.img_ids[index]
    #     img_name = self.imgId2info[img_id]['file_name']
    #     img_path = os.path.join(self.img_dir, img_name)
    #     img = imgUtils.imread_pil(img_path)
    #     # get labels
    #     anns = self.imgId2anns[img_id]
    #     labels = self._ann2labels(anns, img.height, img.width, self.bb_format)
    #     assert labels.masks is not None
    #     # if dataset is not videos, try to generate previous frames
    #     bg_img_dir = self.img_dir + '_np' # background image path
    #     assert os.path.exists(bg_img_dir)
    #     bg_path = os.path.join(bg_img_dir, img_name)
    #     background = imgUtils.imread_pil(bg_path)
    #     # import numpy as np; import matplotlib.pyplot as plt;
    #     # plt.imshow(np.array(img)); plt.show()
    #     # plt.imshow(np.array(background)); plt.show()
    #     t_interval = 1 / self.aug_setting['simulation_fps']
    #     augUtils.random_place(img, labels, background, dt=t_interval)
    #     labels.masks
    #     debug = 1
    #     # augUtils.augment_PIL()
    #     # return (img, labels, img_id, pad_info)

    def _load_single_pil(self, index, to_square=True) -> tuple:
        '''
        One image-label pair for the given index is picked up and pre-processed.
        
        Returns:
            img: PIL.Image
            labels:
            img_id:
            pad_info:
        '''
        # load the image
        img_id = self.img_ids[index]
        imgInfo = self.imgId2info[img_id]
        img_name = imgInfo['file_name']
        img_path = os.path.join(self.img_dir, img_name)
        img = imgUtils.imread_pil(img_path)
        assert imgInfo['height'] == img.height and imgInfo['width'] == img.width
        # get annotations
        anns = self.imgId2anns[img_id]
        labels = self._ann2labels(anns, img.height, img.width, self.bb_format)
        # augmentation
        if self.aug_setting is not None:
            img, labels = augUtils.augment_PIL([img], [labels], self.aug_setting)
            img, labels = img[0], labels[0]
        # pad to square
        aug_flag = (self.aug_setting is not None)
        if to_square:
            img, labels, pad_info = imgUtils.rect_to_square(img, labels,
                self.img_size, aug=aug_flag, resize_step=self.input_divisibility)
        else:
            pad_info = None
        return (img, labels, img_id, pad_info)

    @staticmethod
    def _ann2labels(anns, img_h, img_w, bb_format):
        bboxes = [a['bbox'] for a in anns]
        cat_idxs = [a['cat_idx'] for a in anns]
        if 'rle' not in anns[0]:
            rles = None
        else:
            rles = [a['rle'] for a in anns]
        labels = ImageObjects(
            bboxes=torch.FloatTensor(bboxes),
            cats=torch.LongTensor(cat_idxs),
            masks=None if rles is None else maskUtils.rle2mask(rles),
            bb_format=bb_format,
            img_hw=(img_h, img_w)
        )
        return labels

    @staticmethod
    def collate_func(batch):
        img_batch = torch.stack([items[0] for items in batch])
        label_batch = [items[1] for items in batch]
        ids_batch = [items[2] for items in batch]
        pad_info_batch = [items[3] for items in batch]
        return img_batch, label_batch, ids_batch, pad_info_batch
