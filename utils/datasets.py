import os
import json
import random
from collections import defaultdict
from pycocotools import cocoeval
import PIL.Image
import torch
import torchvision.transforms.functional as tvf

from . import image_ops as imgUtils, augmentation as augUtils
from .structures import ImageObjects
from .evaluation.coco import coco_evaluate_json


def get_trainingset(cfg: dict):
    dataset_name: str = cfg['train.dataset_name']
    if dataset_name in {'COCOtrain2017', 'COCOval2017'}:
        # Official COCO dataset
        _dname = dataset_name.replace('COCO', '')
        training_set_cfg = {
            'img_dir': f'../Datasets/COCO/{_dname}',
            'json_path': f'../Datasets/COCO/annotations/instances_{_dname}.json',
            'ann_bbox_format': 'x1y1wh',
            'is_video': False,
            'noperson_img_dir': f'../Datasets/COCO/{_dname}_np'
        }
        # These datasets are not designed for rotation augmentation
        if cfg['train.data_augmentation'] is not None:
            assert cfg['train.data_augmentation']['rotation'] == False
    elif dataset_name in {'rotbbox_train2017', 'rotbbox_val2017',
                          'personrbb_train2017', 'personrbb_val2017'}:
        # Customized COCO dataset
        _dname = dataset_name.split('_')[1]
        training_set_cfg = {
            'img_dir': f'../Datasets/COCO/{_dname}',
            'json_path': f'../Datasets/COCO/annotations/{dataset_name}.json',
            'ann_bbox_format': 'cxcywhd',
            'is_video': False,
            'noperson_img_dir': f'../Datasets/COCO/{_dname}_np'
        }
        if cfg['train.data_augmentation'] is not None:
            assert cfg['train.data_augmentation']['rotation'] == True
            cfg['train.data_augmentation'].update(rotation_expand=True)
    elif dataset_name in {'debug_zebra', 'debug_kitchen', 'debug3'}:
        training_set_cfg = {
            'img_dir': f'./images/{dataset_name}/',
            'json_path': f'./utils/debug/{dataset_name}.json',
            'ann_bbox_format': 'x1y1wh',
            'is_video': False,
        }
        # These datasets are not designed for rotation augmentation
        assert cfg['train.data_augmentation'] is None
    elif dataset_name == {'rotbb_debug3', 'debug_lunch31'}:
        training_set_cfg = {
            'img_dir': f'./images/{dataset_name}/',
            'json_path': f'./utils/debug/{dataset_name}.json',
            'ann_bbox_format': 'cxcywhd'
        }
        assert cfg['train.data_augmentation'] is None
    else:
        raise NotImplementedError()
    return Dataset4ObjDet(training_set_cfg, cfg)


def get_valset(valset_name):
    if valset_name == 'COCOval2017':
        img_dir = '../Datasets/COCO/val2017'
        val_json_path = '../Datasets/COCO/annotations/instances_val2017.json'
        gt_json = json.load(open(val_json_path, 'r'))
        eval_info = [(os.path.join(img_dir, imi['file_name']), imi['id']) \
                     for imi in gt_json['images']]
        validation_func = lambda x: coco_evaluate_json(x, val_json_path)
    elif valset_name == 'personrbb_val2017':
        img_dir = '../Datasets/COCO/val2017'
        val_json_path = '../Datasets/COCO/annotations/personrbb_val2017.json'
        gt_json = json.load(open(val_json_path, 'r'))
        eval_info = [(os.path.join(img_dir, imi['file_name']), imi['id']) \
                     for imi in gt_json['images']]
        from .evaluation.cepdof import evaluate_json
        validation_func = lambda x: evaluate_json(x, val_json_path)
    elif valset_name in {'Lunch1', 'Lunch2', 'Lunch3', 'Edge_cases',
                        'High_activity', 'All_off', 'IRfilter', 'IRill',
                        'MW-R',
                        'Meeting1', 'Meeting2', 'Lab1', 'Lab2'}:
        img_dir = f'../Datasets/COSSY/frames/{valset_name}'
        val_json_path = f'../Datasets/COSSY/annotations/{valset_name}.json'
        gt_json = json.load(open(val_json_path, 'r'))
        eval_info = [(os.path.join(img_dir, imi['file_name']), imi['id']) \
                     for imi in gt_json['images']]
        from .evaluation.cepdof import evaluate_json
        validation_func = lambda x: evaluate_json(x, val_json_path)
    elif valset_name == 'debug3':
        img_dir = './images/debug3/'
        val_json_path = './utils/debug/debug3.json'
        gt_json = json.load(open(val_json_path, 'r'))
        eval_info = [(os.path.join(img_dir, imi['file_name']), imi['id']) \
                     for imi in gt_json['images']]
        validation_func = lambda x: coco_evaluate_json(x, val_json_path)
    elif valset_name == 'debug_zebra':
        img_dir = './images/debug_zebra/'
        val_json_path = './utils/debug/debug_zebra.json'
        gt_json = json.load(open(val_json_path, 'r'))
        eval_info = [(os.path.join(img_dir, imi['file_name']), imi['id']) \
                     for imi in gt_json['images']]
        validation_func = lambda x: coco_evaluate_json(x, val_json_path)
    elif valset_name == 'debug_kitchen':
        img_dir = './images/debug_kitchen/'
        val_json_path = './utils/debug/debug_kitchen.json'
        gt_json = json.load(open(val_json_path, 'r'))
        eval_info = [(os.path.join(img_dir, imi['file_name']), imi['id']) \
                     for imi in gt_json['images']]
        validation_func = lambda x: coco_evaluate_json(x, val_json_path)
    elif valset_name == 'debug_lunch31':
        img_dir = './images/debug_lunch31/'
        val_json_path = './utils/debug/debug_lunch31.json'
        gt_json = json.load(open(val_json_path, 'r'))
        eval_info = [(os.path.join(img_dir, imi['file_name']), imi['id']) \
                     for imi in gt_json['images']]
        from .evaluation.cepdof import evaluate_json
        validation_func = lambda x: evaluate_json(x, val_json_path)
    elif valset_name == 'rotbb_debug3':
        img_dir = './images/rotbb_debug3/'
        val_json_path = './utils/debug/rotbb_debug3.json'
        gt_json = json.load(open(val_json_path, 'r'))
        eval_info = [(os.path.join(img_dir, imi['file_name']), imi['id']) \
                     for imi in gt_json['images']]
        from .evaluation.cepdof import evaluate_json
        validation_func = lambda x: evaluate_json(x, val_json_path)
    else:
        raise NotImplementedError()
    return eval_info, validation_func


class Dataset4ObjDet(torch.utils.data.Dataset):
    """
    Dataset for training object detection CNNs.

    Args:
        img_dir: str, imgs folder, e.g. 'someDir/COCO/train2017/'
        json_path: str, e.g. 'someDir/COCO/annotations/instances_train2017.json'
        img_size: int, target image size input to the YOLO, default: 608
        augmentation: bool, default: True
        debug: bool, if True, only one data id is selected from the dataset
    """
    def __init__(self, dataset_cfg: dict, global_cfg: dict):
        self.img_dir = dataset_cfg['img_dir']
        self.img_size = global_cfg['train.initial_imgsize']
        self.input_format = global_cfg['general.input_format']
        self.aug_setting = global_cfg['train.data_augmentation']
        self.input_divisibility = global_cfg['general.input_divisibility']

        # Special settings
        self.frame_concat = global_cfg.get('general.input.frame_concatenation', None)
        self.mosaic = self.aug_setting.get('mosaic', False) if (self.aug_setting is not None) else None
        self.is_video = dataset_cfg['is_video'] # TODO:

        self.skip_crowd_ann = True
        self.skip_crowd_img = False
        self.skip_empty_img = True

        self.img_ids = []
        self.imgid2info = dict()
        self.imgid2anns = defaultdict(list)
        self.catid2idx = []
        self.load_json(dataset_cfg['json_path'], dataset_cfg['ann_bbox_format'])

    def load_json(self, json_path, ann_bbox_format):
        '''
        laod json file to self.img_ids, self.imgid2anns
        '''
        assert not self.is_video # TODO:
        print(f'loading annotations {json_path} into memory...')
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        catgys = json_data['categories']
        self.catid2idx = dict([(cat['id'],idx) for idx,cat in enumerate(catgys)])

        if ann_bbox_format in {'x1y1wh', 'cxcywh'}:
            self.bb_format = 'cxcywh'
            self.bb_param = 4
        elif ann_bbox_format == 'cxcywhd':
            self.bb_format = 'cxcywhd'
            self.bb_param = 5
        else:
            raise Exception('Bounding box format is not supported')
        for ann in json_data['annotations']:
            # Parse bounding box annotation
            assert len(ann['bbox']) == self.bb_param
            if self.skip_crowd_ann and ann['iscrowd']:
                continue
            if ann_bbox_format == 'x1y1wh':
                # The dataset is using (x1,y1,w,h). Converting to (cx,cy,w,h)
                ann['bbox'][0] = ann['bbox'][0] + ann['bbox'][2] / 2
                ann['bbox'][1] = ann['bbox'][1] + ann['bbox'][3] / 2
            ann['cat_idx'] = self.catid2idx[ann['category_id']]
            self.imgid2anns[ann['image_id']].append(ann)

        self.imgId2labels = dict()
        for img in json_data['images']:
            img_id = img['id']
            anns = self.imgid2anns[img_id]
            if self.skip_crowd_img and any(ann['iscrowd'] for ann in anns):
                # if there is crowd gt, skip this image
                continue
            if self.skip_empty_img and len(anns) == 0:
                # if there is no object in this image, skip this image
                continue
            self.img_ids.append(img_id)
            self.imgid2info[img['id']] = img
            # convert annotations from json format to ImageObjects format
            bboxes = []
            cat_idxs = []
            for ann in anns:
                bboxes.append(ann['bbox'])
                cat_idxs.append(ann['cat_idx'])
            labels = ImageObjects(
                bboxes=torch.FloatTensor(bboxes),
                cats=torch.LongTensor(cat_idxs),
                bb_format=self.bb_format,
                img_hw=(img['height'], img['width'])
            )
            assert img_id not in self.imgId2labels
            self.imgId2labels[img_id] = labels
        debug = 1

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        """
        Args:
            index (int): data index
        """
        if self.frame_concat is not None:
            img_label_pair = self._load_concat_frames(index, to_square=True)
        elif self.mosaic == True:
            index = random.randint(0, len(self.img_ids)-1)
            pairs = []
            for _ in range(4):
                img_label_pair = self._load_single_pil(index, to_square=False)
                pairs.append(img_label_pair)
            img_label_pair = augUtils.mosaic(pairs, self.img_size)
        else:
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

    def _load_concat_frames(self, index, to_square=True) -> tuple:
        assert not self.is_video # TODO:
        assert self.frame_concat >= 2
        # load the image
        img_id = self.img_ids[index]
        img_name = self.imgid2info[img_id]['file_name']
        img_path = os.path.join(self.img_dir, img_name)
        img = imgUtils.imread_pil(img_path)
        # get labels
        labels = self.imgId2labels[img_id]
        assert isinstance(labels, ImageObjects)
        assert labels.img_hw == (img.height, img.width)
        labels = labels.clone()

        return (img, labels, img_id, pad_info)

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
        img_name = self.imgid2info[img_id]['file_name']
        img_path = os.path.join(self.img_dir, img_name)
        img = imgUtils.imread_pil(img_path)
        # get labels
        labels = self.imgId2labels[img_id]
        assert isinstance(labels, ImageObjects)
        assert labels.img_hw == (img.height, img.width)
        labels = labels.clone()
        # augmentation
        if self.aug_setting is not None:
            img, labels = self.augment_PIL(img, labels)
        # pad to square
        aug_flag = (self.aug_setting is not None)
        if to_square:
            img, labels, pad_info = imgUtils.rect_to_square(img, labels,
                self.img_size, aug=aug_flag, resize_step=self.input_divisibility)
        else:
            pad_info = None
        return (img, labels, img_id, pad_info)
    
    def augment_PIL(self, img, labels):
        # TODO: move this function to augmentation.py
        if torch.rand(1).item() > 0.5:
            low, high = self.aug_setting.get('brightness', [0.6, 1.4])
            img = tvf.adjust_brightness(img, uniform(low, high))
        if torch.rand(1).item() > 0.5:
            low, high = self.aug_setting.get('contrast', [0.5, 1.5])
            img = tvf.adjust_contrast(img, uniform(low, high))
        if torch.rand(1).item() > 0.5:
            low, high = self.aug_setting.get('hue', [-0.1, 0.1])
            img = tvf.adjust_hue(img, uniform(low, high))
        if torch.rand(1).item() > 0.5:
            low, high = self.aug_setting.get('saturation', [0, 2])
            img = tvf.adjust_saturation(img, uniform(low, high)) # 0 ~ 3
        # if torch.rand(1).item() > 0.5:
        #     img = tvf.adjust_gamma(img, uniform(0.5, 3))
        # horizontal flip
        if torch.rand(1).item() > 0.5:
            img, labels = augUtils.hflip(img, labels)
        if self.bb_format in {'cxcywhd'}:
            # vertical flip
            if torch.rand(1).item() > 0.5:
                img, labels = augUtils.vflip(img, labels)
            # random rotation
            rand_deg = torch.rand(1).item() * 360
            expand = self.aug_setting['rotation_expand']
            img, labels = augUtils.rotate(img, rand_deg, labels, expand=expand)
            return img, labels

        return img, labels

    @staticmethod
    def collate_func(batch):
        img_batch = torch.stack([items[0] for items in batch])
        label_batch = [items[1] for items in batch]
        ids_batch = [items[2] for items in batch]
        pad_info_batch = [items[3] for items in batch]
        return img_batch, label_batch, ids_batch, pad_info_batch
    
    def to_dataloader(self, **kwargs):
        return torch.utils.data.DataLoader(self,
                    collate_fn=Dataset4ObjDet.collate_func, **kwargs)


def uniform(a, b):
    return a + torch.rand(1).item() * (b-a)
