from tqdm import tqdm
import json
import cv2
import random

from settings import COSSY_DIR
from utils.visualization import random_colors, _draw_xywha


if __name__ == "__main__":
    ann_data = json.load(open(COSSY_DIR + '/annotations/MW-R_mot.json'))

    nColors = 12
    COLORS = random_colors(num=nColors, dtype='uint8').tolist()

    fw, fh = 768, 768
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out_path = f'./videos_with_ann/Edge_test_mot.avi'
    # vout = cv2.VideoWriter(out_path, fourcc, 10, (fw,fh))

    random.shuffle(ann_data['videos'])
    for i, vidinfo in enumerate(tqdm(ann_data['videos'])):
        id2color = dict()
        vname = vidinfo['id']
        for imname, img_anns in zip(vidinfo['file_names'], vidinfo['annotations']):
            # impath = os.path.join(f'./frames/{vname}/{imname}')
            impath = f'{COSSY_DIR}/frames/{imname}'
            im = cv2.imread(impath)
            object_ids = [ann.get('person_id', None) for ann in img_anns]
            for ann in img_anns:
                assert ann['category_id'] == 1
                cx,cy,w,h,degree = ann['bbox']
                pid = ann.get('person_id', None)
                if pid is None:
                    _clr = (0,255,0)
                else:
                    if pid not in id2color:
                        _avail = set(range(nColors))
                        [_avail.discard(id2color.get(hid,None)) for hid in object_ids]
                        if len(_avail) == 0:
                            print('Warning: colors are exhausted.')
                            id2color[pid] = random.randint(0,nColors-1)
                        else:
                            _avail = list(_avail)
                            id2color[pid] = random.sample(_avail, 1)[0]
                    _clr = id2color[pid]
                    _clr = COLORS[_clr]
                _draw_xywha(im, cx, cy, w, h, degree, color=_clr)
            im = cv2.resize(im, dsize=(fw,fh))
            cv2.imshow('', im)
            cv2.waitKey(1)
    #         vout.write(im)
    # vout.release()
    cv2.destroyAllWindows()