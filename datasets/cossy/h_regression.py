import os
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt


def get_r(bbox, img_hw):
    cx, cy, w, h, a = bbox
    imh, imw = img_hw
    cx = cx - imw/2
    cy = imh/2 - cy
    r = np.sqrt(cx**2 + cy**2)
    return r


def regression(json_path):
    ann_data = json.load(open(json_path, 'r'))
    video_info = ann_data['videos'][0]

    video_hw = (video_info['height'], video_info['width'])
    # build dataset
    annotations = []
    annotations = [annotations.extend(anns) for anns in video_info['annotations']]
        
    for ann in annotations:
        # get 
        raise NotImplementedError()

    debug = 1


if __name__ == "__main__":
    from settings import COSSY_DIR
    # json_names = [s for s in os.listdir('./annotations') if s.endswith('.json')]
    # json_names = ['High_activity.json']
    json_names = ['Edge_cases_mot.json']
    for jsname in json_names:
        jspath = os.path.join(COSSY_DIR, 'annotations', jsname)
        regression(jspath)
    