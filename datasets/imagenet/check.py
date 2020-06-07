import os
import json
from tqdm import tqdm

from settings import ILSVRC_DIR


def check_det(json_path, img_dir):
    if not os.path.exists(json_path):
        print(f'Check failed: JSON file {json_path} does not exist.')
        return False
    ann_data = json.load(open(json_path, 'r'))
    for imgInfo in tqdm(ann_data['images']):
        imname = imgInfo['file_name']
        impath = os.path.join(img_dir, imname)
        if not os.path.exists(impath):
            print(f'Check failed: images of {json_path} does not exist in {img_dir}.')
            return False
    print(f'Check passed: {json_path}')
    return True


def check():
    json_path = f'{ILSVRC_DIR}/Annotations/DET_30_and_VID_every15.json'
    img_dir = f'{ILSVRC_DIR}/Data'
    check_det(json_path, img_dir)

    json_path = f'{ILSVRC_DIR}/Annotations/VID_det_val_2017new.json'
    img_dir = f'{ILSVRC_DIR}/Data/VID/val'
    check_det(json_path, img_dir)


if __name__ == "__main__":
    check()
