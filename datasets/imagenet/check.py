import os
import json
from tqdm import tqdm

from settings import ILSVRC_DIR


def check_det():
    json_path = f'{ILSVRC_DIR}/Annotations/DET_train_30classes.json'
    img_dir = f'{ILSVRC_DIR}/Data/DET'
    
    ann_data = json.load(open(json_path, 'r'))
    for imgInfo in tqdm(ann_data['images']):
        imname = imgInfo['file_name']
        impath = os.path.join(img_dir, imname)
        assert os.path.exists(impath)


if __name__ == "__main__":
    check_det()
