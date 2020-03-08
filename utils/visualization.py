import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from .utils import COCO_CATEGORY_LIST


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
    assert bboxes.dim() == 2 and bboxes.shape[1] >= 5
    line_width = round(im.shape[0] / 200)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = im.shape[0] // 300
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
        cv2.putText(im, text, (int(x1),int(y1)), font, 1,
                    (255,255,255), font_bold, cv2.LINE_AA)
    # plt.imshow(im)
    # plt.show()

# def draw_anns_on_np(im, annotations, draw_angle=False, color=(0,0,255)):
#     '''
#     im: image numpy array, shape(h,w,3), RGB
#     annotations: list of dict, json format
#     '''
#     line_width = im.shape[0] // 500
#     for ann in annotations:
#         x, y, w, h, a = ann['bbox']
#         _draw_xywha(im, x, y, w, h, a, color=color, linewidth=line_width)


# def flow_to_rgb(flow, plt_show=False):
#     '''
#     Visualizing optical flow using a RGB image

#     Args:
#         flow: 2xHxW tensor, flow[0,...] is horizontal motion
#     '''
#     assert torch.is_tensor(flow) and flow.dim() == 3 and flow.shape[0] == 2

#     flow = flow.cpu().numpy()
#     mag, ang = cv2.cartToPolar(flow[0, ...], flow[1, ...], angleInDegrees=True)
#     hsv = np.zeros((flow.shape[1],flow.shape[2],3), dtype=np.uint8)
#     hsv[..., 0] = ang / 2
#     hsv[..., 1] = mag
#     hsv[..., 2] = 255
#     rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#     if plt_show:
#         plt.imshow(rgb)
#         plt.show()
#     return rgb


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
