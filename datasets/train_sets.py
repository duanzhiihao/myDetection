'''Interface between train.py and datasets'''


def get_trainingset(cfg: dict):
    dataset_name: str = cfg['train.dataset_name']
    # ------------------------ image datasets ------------------------
    if dataset_name in {'COCOtrain2017', 'COCOval2017'}:
        # Official COCO dataset
        from settings import COCO_DIR
        split_name = dataset_name.replace('COCO', '')
        training_set_cfg = {
            'img_dir': f'{COCO_DIR}/{split_name}',
            'ann_path': f'{COCO_DIR}/annotations/instances_{split_name}.json',
            'ann_bbox_format': 'x1y1wh',
        }

        # These datasets are not designed for rotation augmentation
        if cfg['train.data_augmentation'] is not None:
            assert cfg['train.data_augmentation']['rotation'] == False

        from .image_dataset import ImageDataset
        return ImageDataset(training_set_cfg, cfg)

    elif dataset_name in {'rotbbox_train2017', 'rotbbox_val2017',
                          'personrbb_train2017', 'personrbb_val2017'}:
        # Customized COCO dataset
        from settings import COCO_DIR
        split_name = dataset_name.split('_')[1]
        training_set_cfg = {
            'img_dir': f'{COCO_DIR}/{split_name}',
            'ann_path': f'{COCO_DIR}/annotations/{dataset_name}.json',
            'ann_bbox_format': 'cxcywhd',
        }

        if cfg['train.data_augmentation'] is not None:
            assert cfg['train.data_augmentation']['rotation'] == True
            cfg['train.data_augmentation'].update(rotation_expand=True)

        from .image_dataset import ImageDataset
        return ImageDataset(training_set_cfg, cfg)

    # ------------------------ video datasets ------------------------
    elif dataset_name in {'HBMWR_mot_train'}:
        # COSSY multi object tracking dataset
        from settings import COSSY_DIR
        training_set_cfg = {
            'img_dir': f'{COSSY_DIR}/frames',
            'ann_path': f'{COSSY_DIR}/annotations/{dataset_name}.json',
            'ann_bbox_format': 'cxcywhd',
            'static_background': True
        }
        if cfg['train.data_augmentation.clip'] is not None:
            assert cfg['train.data_augmentation.clip']['rotation'] == True
            cfg['train.data_augmentation.clip'].update(rotation_expand=False)
        from .video_dataset import Dataset4VODT
        return Dataset4VODT(training_set_cfg, cfg)
    
    # ------------------------ datasets for debugging ------------------------
    elif dataset_name in {'debug_zebra', 'debug_kitchen', 'debug3'}:
        from settings import PROJECT_ROOT
        training_set_cfg = {
            'img_dir': f'{PROJECT_ROOT}/images/{dataset_name}/',
            'ann_path': f'{PROJECT_ROOT}/datasets/debug/{dataset_name}.json',
            'ann_bbox_format': 'x1y1wh',
        }
        # These datasets are not designed for rotation augmentation
        from .image_dataset import ImageDataset
        return ImageDataset(training_set_cfg, cfg)
    elif dataset_name in {'rotbb_debug3', 'debug_lunch31', 'rot80_debug1'}:
        from settings import PROJECT_ROOT
        training_set_cfg = {
            'img_dir': f'{PROJECT_ROOT}/images/{dataset_name}/',
            'ann_path': f'{PROJECT_ROOT}/datasets/debug/{dataset_name}.json',
            'ann_bbox_format': 'cxcywhd'
        }
        assert cfg['train.data_augmentation'] is None
        from .image_dataset import ImageDataset
        return ImageDataset(training_set_cfg, cfg)

    # --------------------------------- end ---------------------------------
    else:
        raise NotImplementedError()
