from typing import List
import os
import json
import random
import PIL.Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tvf

import utils.mask_ops as maskUtils
import utils.image_ops as imgUtils
import utils.augmentation as augUtils
from utils.structures import ImageObjects


class Dataset4VODT(Dataset):
    """
    Dataset for training video object detection and tracking algorithms.
    """
    def __init__(self, dataset_cfg, global_cfg: dict):
        self.img_dir            = dataset_cfg['img_dir']
        self.seq_len            = global_cfg['train.sequence_length']
        self.img_size           = global_cfg['train.initial_imgsize']
        self.input_format       = global_cfg['general.input_format']
        self.clip_aug           = global_cfg['train.data_augmentation.clip']
        self.frame_aug          = global_cfg['train.data_augmentation.frame']
        self.input_divisibility = global_cfg['general.input_divisibility']

        # special setting
        self.mosaic = self.clip_aug['mosaic']

        # if background is static, we can (almost) safely skip empty images
        self.skip_empty_img = dataset_cfg['static_background']

        self.frame_paths  = []
        self.annotations  = []
        self.video_ids    = []
        self.videoId2info = dict()
        self.catId2idx    = dict()
        self._load_json(dataset_cfg['ann_path'], dataset_cfg['ann_bbox_format'])

    def _load_json(self, json_path, ann_bbox_format):
        '''load json file'''
        print(f'Loading annotations {json_path} into memory...')
        with open(json_path, 'r') as f:
            ann_data: dict = json.load(f)

        # categories
        for idx, cat in enumerate(ann_data['categories']):
            self.catId2idx[cat['id']] = idx

        # set bounding box format
        if ann_bbox_format == 'cxcywhd':
            self.bb_format = 'cxcywhd'
            self.bb_param = 5
        else:
            raise Exception('Bounding box format is not supported')

        # traverse over all images
        video_start = []
        for video in ann_data['videos']:
            video: dict
            assert video['id'] not in self.videoId2info, 'Two video have the same id'
            self.videoId2info[video['id']] = video
            vidh, vidw = video['height'], video['width']
            video_start.append(True)
            for imname, img_anns in zip(video['file_names'], video['annotations']):
                impath = os.path.join(self.img_dir, imname)
                assert os.path.exists(impath)
                # skip empty image
                if self.skip_empty_img and len(img_anns) == 0:
                    continue
                for ann in img_anns:
                    assert len(ann['bbox']) == self.bb_param
                    # convert bbox format if necessary
                    if ann_bbox_format == 'x1y1wh':
                        ann['bbox'][0] = ann['bbox'][0] + ann['bbox'][2] / 2
                        ann['bbox'][1] = ann['bbox'][1] + ann['bbox'][3] / 2
                    # category inddex
                    ann['cat_idx'] = self.catId2idx[ann['category_id']]
                    if ann['segmentation'] != []:
                        segm = ann.pop('segmentation')
                        ann['rle'] = maskUtils.segm2rle(segm, vidh, vidw)
                self.frame_paths.append(impath)
                self.annotations.append(img_anns)
                video_start.append(False)
                self.video_ids.append(video['id'])
            video_start.pop(-1)
        self.video_start = torch.BoolTensor(video_start)

    def __len__(self):
        return 1000000000

    def __getitem__(self, _):
        assert len(self.frame_paths) == len(self.annotations) == len(self.video_start)
        if self.mosaic == True:
            raise NotImplementedError()
            # index = random.randint(0, len(self.img_ids)-1)
            # pairs = []
            # for _ in range(4):
            #     img_label_pair = self._load_single_pil(index, to_square=False)
            #     pairs.append(img_label_pair)
            # img_label_pair = augUtils.mosaic(pairs, self.img_size)
        else:
            img_label_pairs = self._load_random_seq(to_square=True)
        
        seq_imgs, seq_labels, start_flags, img_paths, pad_info = img_label_pairs
        assert len(seq_imgs) == len(seq_labels) == len(start_flags) \
               == len(img_paths) == len(pad_info)
        # visualize the clip for debugging
        if False:
            import numpy as np; import cv2
            for _im, _lab in zip(seq_imgs, seq_labels):
                _im = np.array(_im)
                _lab: ImageObjects
                _lab.draw_on_np(_im)
                cv2.imshow('', _im)
                cv2.waitKey(1000)

        # Convert PIL.image to torch.Tensor with shape (3,h,w) if it's not
        if isinstance(seq_imgs[0], PIL.Image.Image):
            seq_imgs = [tvf.to_tensor(img) for img in seq_imgs]
        assert all([isinstance(img, torch.FloatTensor) for img in seq_imgs])
        assert all([isinstance(_lab, ImageObjects) for _lab in seq_labels])
        # Noise augmentation
        if self.frame_aug is not None:
            # blur = [augUtils.random_avg_filter, augUtils.max_filter,
            #         augUtils.random_gaussian_filter]
            # if torch.rand(1).item() > 0.7:
            #     blur_func = random.choice(blur)
            #     img = blur_func(img)
            p = self.frame_aug['satpepper_noise_density']
            for img in seq_imgs:
                if torch.rand(1).item() > 0.7:
                    augUtils.add_saltpepper(img, max_p=p)
        # Convert into desired input format, e.g., normalized
        seq_imgs = [imgUtils.format_tensor_img(img, code=self.input_format) \
                    for img in seq_imgs]
        # Remove annotations which are too small
        seq_labels = [_lab[_lab.bboxes[:,2] * _lab.bboxes[:,3] >= 32] \
                      for _lab in seq_labels]

        # sanity check before return
        for _lab in seq_labels:
            _lab.bboxes[:, 0:4].clamp_(min=0)
        return seq_imgs, seq_labels, start_flags, tuple(img_paths)

    def _load_random_seq(self, to_square=True):
        '''Load a sequence of images and labels'''
        # get a random batch
        index = random.randint(0, len(self.video_start)-self.seq_len)
        seq_imgs = []
        seq_labels = []
        start_flags = []
        img_paths = []
        while len(seq_imgs) < self.seq_len and index < len(self.video_start):
            vinfo = self.videoId2info[self.video_ids[index]]
            impath = self.frame_paths[index]
            img = imgUtils.imread_pil(impath)
            assert img.height == vinfo['height'] and img.width == vinfo['width']
            anns = self.annotations[index]
            labels = self._ann2labels(anns, img.height, img.width, self.bb_format)
            seq_imgs.append(img)
            seq_labels.append(labels)
            start_flags.append(self.video_start[index])
            img_paths.append(impath)
            # temporal down sampling
            if 'min_fps' in self.clip_aug:
                max_step = round(vinfo['fps']/self.clip_aug['min_fps'])
            else:
                max_step = 1
            index += random.randint(1, max(max_step,1))
        start_flags[0] = torch.BoolTensor([True])[0] # the first image is always start
        # augmentation
        if self.clip_aug is not None:
            seq_imgs, seq_labels = augUtils.augment_PIL(seq_imgs, seq_labels,
                                                        self.clip_aug)
        # pad to square
        aug_flag = self.clip_aug['resize']
        if to_square:
            seq_imgs, labels, pad_info = imgUtils.rect_to_square(seq_imgs, seq_labels,
                    self.img_size, aug=aug_flag, resize_step=self.input_divisibility)
        else:
            pad_info = None
        return (seq_imgs, labels, start_flags, img_paths, pad_info)

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
        _seqs, _labels, _flags, _ids = [list(b) for b in zip(*batch)]
        seq_imgs:   List[torch.tensor] = [torch.stack(b) for b in zip(*_seqs)]
        seq_labels: List[list]         = [list(b) for b in zip(*_labels)]
        seq_flags:  List[torch.tensor] = [torch.stack(b) for b in zip(*_flags)]
        img_ids:    List[tuple]        = [list(b) for b in zip(*_ids)]
        return seq_imgs, seq_labels, seq_flags, img_ids
    
    def to_iter(self, **kwargs):
        self.iter = iter(DataLoader(self, collate_fn=self.collate_func, **kwargs))

    def get_next(self):
        assert hasattr(self, 'to_iter'), 'Please call to_iter() first'
        data = next(self.iter)
        return data
