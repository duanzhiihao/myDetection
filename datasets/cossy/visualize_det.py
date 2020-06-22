import os
import json
from random import choices
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
from utils.visualization import _draw_xywha
from settings import COSSY_DIR


json_path = f'{COSSY_DIR}/annotations/youtube_val.json'
img_dir = f'{COSSY_DIR}/frames/'
# img_dir = '../COCO/val2017/'


json_data = json.load(open(json_path, 'r'))
imgid2anns = defaultdict(list)
for ann in json_data['annotations']:
    img_id = ann['image_id']
    imgid2anns[img_id].append(ann)

# for imgInfo in choices(json_data['images'], k=100):
for imgInfo in json_data['images']:
    fname = imgInfo['file_name']
    img_path = os.path.join(img_dir, fname)
    im = plt.imread(img_path)
    
    anns = imgid2anns[imgInfo['id']]
    for ann in anns:
        # Get box properties
        x,y,w,h,angle = ann['bbox']
        assert angle >= -90 and angle < 90
        _draw_xywha(im, x, y, w, h, angle, color=(0,255,0), linewidth=3)
        text = str(ann['person_id'])
        cv2.putText(im, text, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255,255,255), 2, cv2.LINE_AA)
    
    plt.figure()
    plt.imshow(im)
    plt.show()
    # cv2.imwrite('MW_example.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    # fig.savefig('out.jpg', bbox_inches='tight', pad_inches=0)
    debug = 1
