import os
import json
from tqdm import tqdm
import random
import xml.etree.ElementTree as ElementTree
import cv2
import matplotlib.pyplot as plt

from settings import ILSVRC_DIR, PROJECT_ROOT
from utils.visualization import _draw_xywha
from class_map import catId_to_name


def visualize_xml():
    imgset_path = f'{PROJECT_ROOT}/datasets/imagenet/VID_val_2017new.txt'
    task = 'VID'
    split = 'val'

    assert os.path.exists(imgset_path)
    img_names = open(imgset_path, 'r').read().strip().split('\n')

    ignored = 0
    count = 0
    # for iminfo in random.choices(img_names, k=10):
    for iminfo in tqdm(img_names):
        imgId = iminfo.split()[0]
        impath = f'{ILSVRC_DIR}/Data/{task}/{split}/{imgId}.JPEG'
        im = cv2.imread(impath)

        xml_path = f'{ILSVRC_DIR}/Annotations/{task}/{split}/{imgId}.xml'
        # if not os.path.exists(xml_path):
        #     plt.imshow(im[:,:,::-1]); plt.show()
        xml_tree = ElementTree.parse(xml_path)
        objects = xml_tree.findall('object')
        for obj in objects:
            count += 1
            catId = obj.find('name').text
            if catId not in catId_to_name:
                ignored += 1
                continue
            cat_name = catId_to_name[catId]
            # Get box properties
            box = obj.find('bndbox')
            xmin = float(box.find('xmin').text)
            ymin = float(box.find('ymin').text)
            xmax = float(box.find('xmax').text)
            ymax = float(box.find('ymax').text)
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin
            # _draw_xywha(im, cx, cy, w, h, 0)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(im, cat_name, (int(xmin),int(ymin)), font, 0.8,
            #             (255,255,255), 2, cv2.LINE_AA)

        # plt.imshow(im[:,:,::-1]); plt.show()
        debug = 1
    print('total objects:', count)
    print('ignored objects:', ignored)


if __name__ == "__main__":
    visualize_xml()
