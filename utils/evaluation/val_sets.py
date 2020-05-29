import os
import json


def get_valset(valset_name: str):
    '''
    Get validation set.

    Args:
        valset_name: name of the validation set. Available names can be found below.
    '''
    # ------------------------ image datasets ------------------------
    if valset_name == 'COCOval2017':
        from settings import COCO_DIR
        from .coco import coco_evaluate_bbox
        val_json_path = f'{COCO_DIR}/annotations/instances_val2017.json'
        ann_data = json.load(open(val_json_path, 'r'))
        ann_data.pop('annotations')
        eval_info = {
            'image_dir': f'{COCO_DIR}/val2017',
            'image_info': ann_data,
            'eval_type': 'x1y1wh',
            'val_func': coco_evaluate_bbox
        }
        validation_func = lambda x: coco_evaluate_bbox(x, val_json_path)

    elif valset_name == 'personrbb_val2017':
        raise NotImplementedError()
        from settings import COCO_DIR
        img_dir = f'{COCO_DIR}/val2017'
        val_json_path = f'{COCO_DIR}/annotations/personrbb_val2017.json'

        gt_json = json.load(open(val_json_path, 'r'))
        eval_info = [(os.path.join(img_dir, imi['file_name']), imi['id']) \
                     for imi in gt_json['images']]

        from .cepdof import evaluate_json
        validation_func = lambda x: evaluate_json(x, val_json_path)

    elif valset_name in {'Lunch1', 'Lunch2', 'Lunch3', 'Edge_cases',
                         'High_activity', 'All_off', 'IRfilter', 'IRill',
                         'Meeting1', 'Meeting2', 'Lab1', 'Lab2',
                         'MW-R'}:
        raise NotImplementedError()
        from settings import COSSY_DIR
        img_dir = f'{COSSY_DIR}/frames/{valset_name}'
        val_json_path = f'{COSSY_DIR}/annotations/{valset_name}.json'

        gt_json = json.load(open(val_json_path, 'r'))
        eval_info = [(os.path.join(img_dir, imi['file_name']), imi['id']) \
                     for imi in gt_json['images']]

        from .cepdof import evaluate_json
        validation_func = lambda x: evaluate_json(x, val_json_path)

    # ------------------------ video datasets ------------------------
    elif valset_name in {'Lab1_mot',
                         'Lunch2_mot', 'Edge_cases_mot', 'High_activity_mot',
                         'All_off_mot', 'IRfilter_mot', 'IRill_mot'}:
        raise NotImplementedError()
        from settings import COSSY_DIR
        val_json_path = f'{COSSY_DIR}/annotations/{valset_name}.json'
        gt_json = json.load(open(val_json_path, 'r'))
        for vid in gt_json['videos']:
            vid['annotations'] = []
        eval_info = {
            'image_dir': f'{COSSY_DIR}/frames',
            'eval_json': gt_json,
            'eval_type': 'cxcywhd'
        }
        from .cepdof import evaluate_json
        validation_func = lambda x: evaluate_json(x, val_json_path.replace('_mot',''))

    # ------------------------ datasets for debugging ------------------------
    elif valset_name in {'debug_zebra', 'debug_kitchen', 'debug3'}:
        from settings import PROJECT_ROOT
        from .coco import coco_evaluate_bbox
        val_json_path = f'{PROJECT_ROOT}/datasets/debug/{valset_name}.json'
        ann_data = json.load(open(val_json_path, 'r'))
        ann_data.pop('annotations')
        eval_info = {
            'image_dir': f'{PROJECT_ROOT}/images/{valset_name}/',
            'image_info': ann_data,
            'eval_type': 'x1y1wh',
            'val_func': coco_evaluate_bbox
        }
        validation_func = lambda x: coco_evaluate_bbox(x, val_json_path)
    elif valset_name in {'rotbb_debug3', 'debug_lunch31'}:
        raise NotImplementedError()
        from settings import PROJECT_ROOT
        img_dir = f'{PROJECT_ROOT}/images/debug_lunch31/'
        val_json_path = f'{PROJECT_ROOT}/datasets/debug/debug_lunch31.json'
        gt_json = json.load(open(val_json_path, 'r'))
        eval_info = [(os.path.join(img_dir, imi['file_name']), imi['id']) \
                     for imi in gt_json['images']]
        from .cepdof import evaluate_json
        validation_func = lambda x: evaluate_json(x, val_json_path)
        
    else:
        raise NotImplementedError('Unknown validation set name')
    
    return eval_info, validation_func


def get_val_func(validation_code: str='x1y1wh'):
    '''
    Args:
        validation_code
    '''
    raise DeprecationWarning()
    if validation_code == 'x1y1wh':
        # Axis-aligned bounding box
        from .coco import coco_evaluate_bbox
        return coco_evaluate_bbox
    elif validation_code == 'cxcywhd':
        # Rotated bounding box
        raise NotImplementedError()
    elif validation_code == 'mask':
        # Segmentation mask
        raise NotImplementedError()
