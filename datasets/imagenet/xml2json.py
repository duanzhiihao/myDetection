import os
import json
from tqdm import tqdm
import xml.etree.ElementTree as ElementTree
import PIL.Image

from settings import ILSVRC_DIR, PROJECT_ROOT
from class_map import class_ids, catId_to_name

if __name__ == "__main__":
    name = 'DET_train_30classes'
    imgset_path = f'{name}.txt'
    assert os.path.exists(imgset_path)
    img_names = open(imgset_path, 'r').read().strip().split('\n')

    ann_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    # categories
    catcode2id = dict()
    for ci, cat_code in enumerate(class_ids):
        catId = ci + 1
        cat = {
            'id': catId,
            'name': catId_to_name[cat_code]
        }
        ann_data['categories'].append(cat)
        assert cat_code not in catcode2id
        catcode2id[cat_code] = catId
    
    count = 0
    # images and annotations
    for i, iminfo in enumerate(tqdm(img_names)):
        imname = iminfo.split()[0]
        if imname.startswith('train/'):
            imname = imname[6:]
        impath = f'{ILSVRC_DIR}/Data/DET/train/{imname}.JPEG'
        img = PIL.Image.open(impath)

        # add image
        imgId = i
        imgInfo = {
            'file_name': f'{imname}.JPEG',
            'height': img.height,
            'width': img.width,
            'id': imgId
        }
        ann_data['images'].append(imgInfo)

        # read xml annotations
        xml_path = f'{ILSVRC_DIR}/Annotations/DET/train/{imname}.xml'
        xml_tree = ElementTree.parse(xml_path)
        imgw = int(xml_tree.find('size').find('width').text)
        imgh = int(xml_tree.find('size').find('height').text)
        assert imgw == img.width and imgh == img.height

        # convert to coco json annotations
        objects = xml_tree.findall('object')
        for obj in objects:
            count += 1
            cat_code = obj.find('name').text
            if cat_code not in catId_to_name:
                continue
            # cat_name = catId_to_name[cat_code]
            # Get box properties
            box = obj.find('bndbox')
            xmin = float(box.find('xmin').text)
            ymin = float(box.find('ymin').text)
            xmax = float(box.find('xmax').text)
            ymax = float(box.find('ymax').text)
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            x1 = round(xmin, 2)
            y1 = round(ymin, 2)
            w = round(xmax - xmin, 2)
            h = round(ymax - ymin, 2)
            ann = {
                'area': round(w*h),
                'iscrowd': 0,
                'image_id': imgId,
                'bbox': [x1, y1, w, h],
                'segmentation': [],
                'category_id': catcode2id[cat_code],
                'id': count
            }
            ann_data['annotations'].append(ann)

    debug = 1
    save_path = f'{ILSVRC_DIR}/Annotations/{name}.json'
    json.dump(ann_data, open(save_path, 'w'), indent=1)
