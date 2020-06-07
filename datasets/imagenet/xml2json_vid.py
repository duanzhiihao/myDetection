import os
import json
from tqdm import tqdm
import xml.etree.ElementTree as ElementTree
import PIL.Image

from settings import ILSVRC_DIR, PROJECT_ROOT
from class_map import class_ids, catId_to_name


def txts_to_json(txt_paths, skip_frames):
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
    
    img_count = 0
    ann_count = 0
    assert isinstance(txt_paths, list)
    video_names = []
    video_names_set = set()
    for txtpath in txt_paths:
        assert os.path.exists(txtpath)
        candidates = open(txtpath, 'r').read().strip().split('\n')
        names = []
        [names.append(s) for s in candidates if s not in video_names_set]
        video_names.extend(names)
        [video_names_set.add(s) for s in names]
    video_names.sort()

    for vname in tqdm(video_names):
        assert len(vname.split()) == 2
        vname = vname.split()[0]
        img_dir = f'{ILSVRC_DIR}/Data/VID/val/{vname}'
        img_names = os.listdir(img_dir)
        img_names.sort()

        for i, imname in enumerate(img_names):
            if i % skip_frames != 0:
                continue
            img_count += 1
            impath = os.path.join(img_dir, imname)
            img = PIL.Image.open(impath)

            # add image
            imgId = img_count
            file_name = f'{vname}/{imname}'
            imgInfo = {
                'file_name': file_name,
                'height': img.height,
                'width': img.width,
                'id': imgId
            }
            ann_data['images'].append(imgInfo)

            # read xml annotations
            xmlname = imname.replace('.JPEG', '.xml')
            xml_path = f'{ILSVRC_DIR}/Annotations/VID/val/{vname}/{xmlname}'
            xml_tree = ElementTree.parse(xml_path)
            imgw = int(xml_tree.find('size').find('width').text)
            imgh = int(xml_tree.find('size').find('height').text)
            assert imgw == img.width and imgh == img.height

            # convert to coco json annotations
            objects = xml_tree.findall('object')
            for obj in objects:
                ann_count += 1
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
                    'id': ann_count
                }
                ann_data['annotations'].append(ann)

    debug = 1
    save_path = f'{ILSVRC_DIR}/Annotations/VID_det_val_2017new_every{skip_frames}.json'
    json.dump(ann_data, open(save_path, 'w'), indent=1)


if __name__ == "__main__":
    txt_dir = f'{ILSVRC_DIR}/ImageSets/VID'
    txt_names = os.listdir(txt_dir)
    txt_names = [os.path.join(txt_dir, s) for s in txt_names if 'train' in s]
    txt_names.sort()

    txts_to_json(txt_names, skip_frames=15)
