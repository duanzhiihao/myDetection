import os
import json
from random import choices
from collections import defaultdict
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.visualization import _draw_xywha
from settings import COCO_DIR, ILSVRC_DIR

# json_path = f'{COCO_DIR}/annotations/instances_train2017.json'
# img_dir = f'{COCO_DIR}/train2017/'
# img_dir = '../COCO/val2017/'
json_path = f'{ILSVRC_DIR}/Annotations/VID_val_2017new.json'
img_dir = f'{ILSVRC_DIR}/Data/VID/val'


json_data = json.load(open(json_path, 'r'))

img_ids = []
imgid2anns = defaultdict(list)
for img in json_data['images']:
    # if img['id'] != 79841:
    #     continue
    img_ids.append((img['id'], img['file_name']))
for ann in json_data['annotations']:
    img_id = ann['image_id']
    imgid2anns[img_id].append(ann)

catId2name = dict()
for cat in json_data['categories']:
    assert cat['id'] not in catId2name
    catId2name[cat['id']] = cat['name']

for (img_id, imname) in choices(img_ids, k=10):
    img_path = os.path.join(img_dir, imname)
    im = plt.imread(img_path)
    
    anns = imgid2anns[img_id]

    is_crowd = any([ann['iscrowd'] for ann in anns])
    # if not is_crowd:
    #     continue
    for ann in anns:
        # if not ann['iscrowd']: continue
        # Get box properties
        x1,y1,w,h = ann['bbox']
        cx = x1 + w/2
        cy = y1 + h/2
        _draw_xywha(im, cx, cy, w, h, 0, color=(0,255,0), linewidth=2)
        text = catId2name[ann['category_id']]
        cv2.putText(im, text, (int(x1),int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 1, cv2.LINE_AA)
        print(text)
    
    plt.figure(figsize=(8,8))
    plt.imshow(im)
    plt.show()
    # cv2.imwrite('MW_example.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    # fig.savefig('out.jpg', bbox_inches='tight', pad_inches=0)
    debug = 1
