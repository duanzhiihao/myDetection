import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from .constants import COCO_CATEGORY_LIST


def _draw_xywha(im, x, y, w, h, angle, color=(255,0,0), linewidth=5):
    '''
    im: image numpy array, shape(h,w,3), RGB
    angle: degree
    '''
    c, s = np.cos(angle/180*np.pi), np.sin(angle/180*np.pi)
    R = np.asarray([[c, s], [-s, c]])
    pts = np.asarray([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    rot_pts = []
    for pt in pts:
        rot_pts.append(([x, y] + pt @ R).astype(int))
    contours = np.array([rot_pts[0], rot_pts[1], rot_pts[2], rot_pts[3]])
    cv2.polylines(im, [contours], isClosed=True, color=color,
                thickness=linewidth, lineType=cv2.LINE_4)


def draw_cocobb_on_np(im, bboxes, bb_type='pbb', print_dt=False):
    '''
    im: numpy array, uint8, shape(h,w,3), RGB
    bboxes: rows of [class,x,y,w,h,conf]
    '''
    raise DeprecationWarning()

    assert bboxes.dim() == 2 and bboxes.shape[1] >= 5
    line_width = round(im.shape[0] / 300)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = im.shape[0] * im.shape[1] / (700*700)
    font_bold = im.shape[0] // 400
    for bb in bboxes:
        if bb_type == 'gtbb':
            # ground truth BB
            cat_idx,x,y,w,h = bb
            a = 0
            conf = -1
        elif bb_type == 'pbb':
            # predicted BB
            cat_idx,x,y,w,h,conf = bb
            a = 0
        elif bb_type == 'gtrbb':
            # ground truth rotated BB
            cat_idx,x,y,w,h,a, = bb
            conf = -1
        elif bb_type == 'prbb':
            # predicted rotated BB
            cat_idx,x,y,w,h,a,conf = bb
        else:
            raise NotImplementedError()
        cat_idx = int(cat_idx)
        cat_name = COCO_CATEGORY_LIST[cat_idx]['name']
        cat_color = COCO_CATEGORY_LIST[cat_idx]['color']
        if print_dt:
            print(f'category:{cat_name}, score: {conf},',
                  f'[{x:.1f} {y:.1f} {w:.1f} {h:.1f} {a:.1f}].')
        _draw_xywha(im, x, y, w, h, a, color=cat_color, linewidth=line_width)
        x1, y1 = x - w/2, y - h/2
        text = cat_name if 'gt' in bb_type else f'{cat_name}, {conf:.2f}'
        cv2.putText(im, text, (int(x1),int(y1)), font, 0.5,
                    (255,255,255), font_bold, cv2.LINE_AA)
    # plt.imshow(im)
    # plt.show()


def draw_bboxes_on_np(im, img_objs, class_map='COCO', **kwargs):
    print_dt = kwargs.get('print_dt', False)
    # Extract bboxes, scores, and category indices
    obj_num = img_objs.bboxes.shape[0]
    if obj_num == 0: return
    bboxes = img_objs.bboxes
    scores = img_objs.scores if img_objs.scores is not None else \
             [None for _ in range(obj_num)]
    class_indices = img_objs.cats
    assert len(bboxes) == len(scores) == len(class_indices)
    # Select the target dataset
    if class_map == 'COCO':
        cat_idx2name = lambda x: COCO_CATEGORY_LIST[x]['name']
        cat_idx2color = lambda x: COCO_CATEGORY_LIST[x]['color']
    else: raise NotImplementedError()
    # Initialize some drawing parameters
    line_width = round(im.shape[0] / 300)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = im.shape[0] * im.shape[1] / (700*700)
    font_bold = im.shape[0] // 400
    # Interate over all bounding boxes
    for bb, conf, c_idx in zip(bboxes, scores, class_indices):
        if img_objs._bb_format == 'cxcywh':
            cx, cy, w, h = bb
            a = 0
        elif img_objs._bb_format == 'cxcywhd':
            cx, cy, w, h, a = bb
        else: raise NotImplementedError()
        cat_name = cat_idx2name(c_idx)
        cat_color = cat_idx2color(c_idx)
        if print_dt:
            print(f'category:{cat_name}, score: {conf},',
                  f'[{cx:.1f} {cy:.1f} {w:.1f} {h:.1f} {a:.1f}].')
        _draw_xywha(im, cx, cy, w, h, a, color=cat_color, linewidth=line_width)
        x1, y1 = cx - w/2, cy - h/2
        text = cat_name if conf is None else f'{cat_name}, {conf:.2f}'
        cv2.putText(im, text, (int(x1),int(y1)), font, 0.5,
                    (255,255,255), font_bold, cv2.LINE_AA)
    if kwargs.get('imshow', False):
        plt.imshow(im)
        plt.show()


def tensor_to_npimg(tensor_img):
    tensor_img = tensor_img.squeeze()
    assert tensor_img.shape[0] == 3 and tensor_img.dim() == 3
    return tensor_img.permute(1,2,0).cpu().numpy()


def imshow_tensor(tensor_batch):
    batch = tensor_batch.clone().detach().cpu()
    if batch.dim() == 3:
        batch = batch.unsqueeze(0)
    for tensor_img in batch:
        np_img = tensor_to_npimg(tensor_img)
        plt.imshow(np_img)
    plt.show()


def plt_show(im):
    plt.imshow(im)
    plt.show()
