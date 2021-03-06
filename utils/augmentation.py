from typing import List
import random
import numpy as np
import scipy.ndimage
import PIL.Image
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as tvf
import torch.nn.functional as tnf

from .structures import ImageObjects
from .bbox_ops import cxcywh_to_x1y1x2y2


def mosaic(img_label_pairs, target_size:int):
    '''
    Mosaic augmentation as described in YOLOv4: https://arxiv.org/abs/2004.10934
    '''
    assert len(img_label_pairs) == 4
    np_imgs = [cv2.cvtColor(np.array(p[0]), cv2.COLOR_RGB2BGR) \
               for p in img_label_pairs]
    labels  = [p[1] for p in img_label_pairs]
    imgIds  = [p[2] for p in img_label_pairs]

    f = 0.1 # factor
    # mosaic center x and center y
    mcx = int(uniform(target_size * (0.5-f), target_size * (0.5+f)))
    mcy = int(uniform(target_size * (0.5-f), target_size * (0.5+f)))

    new_im = np.zeros((target_size,target_size,3), dtype='float32')
    # top-left image
    im, labels = _crop_img_labels(np_imgs[0], labels[0], (mcy,mcx))
    new_im[0:mcx, 0:mcy, :] = im
    raise NotImplementedError()

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,8)); plt.imshow(new_im); plt.show()
    debug = 1
    new_im = torch.from_numpy(new_im).permute(2, 0, 1)
    return (new_im, labels, imgIds, None)


def _crop_img_labels(im, labels, target_hw):
    '''
    Crop image and labels for mosaic augmentation
    '''
    if isinstance(im, PIL.Image.Image):
        raise NotImplementedError()
        im = np.array(im, dtype='float32') / 255
    assert im.ndim == 3 and im.shape[2] == 3

    tgt_h, tgt_w = target_hw
    # resize the image such that it can fill the target_hw window
    ratio = max(tgt_h/im.shape[0], tgt_w/im.shape[1])
    rh, rw = int(im.shape[0] * ratio), int(im.shape[1] * ratio)
    im = cv2.resize(im, (rw,rh))
    # sanity check
    if rh == tgt_h:
        assert rw >= tgt_w
    else:
        assert (rw == tgt_w) and (rh > tgt_h)
    # random extract window
    hstart = random.randint(0, rh - tgt_h)
    wstart = random.randint(0, rw - tgt_w)
    im = im[hstart:hstart+tgt_h, wstart:wstart+tgt_w, :]

    # modify labels
    labels: ImageObjects
    assert labels._bb_format in {'cxcywh', 'cxcywhd'}
    if labels.masks is None:
        raise NotImplementedError()
        # bounding box only
        bboxes = labels.bboxes
        bboxes[:, 0:4] *= ratio
        bboxes[:, 0] -= wstart
        bboxes[:, 1] -= hstart
        # convert to x1y1x2y2
        bboxes = cxcywh_to_x1y1x2y2(bboxes)
        bboxes.clamp_(min=0)
        bboxes[:, 0].clamp_(max=tgt_w)
    else:
        new_masks = torch.zeros(len(labels), tgt_h, tgt_w)
        for i, mask in enumerate(labels.masks):
            npm = mask.numpy().astype('uint8')
            # plt.imshow(npm, cmap='gray'); plt.show()
            npm = cv2.resize(npm, (rw,rh), interpolation=cv2.INTER_NEAREST)
            npm = npm.astype('bool')
            # plt.imshow(npm, cmap='gray'); plt.show()
            mask = torch.from_numpy(npm)
            mask = mask[hstart:hstart+tgt_h, wstart:wstart+tgt_w]
            new_masks[i] = mask
        labels.masks = new_masks
        labels.mask_to_bbox_()
    # need mask
    raise NotImplementedError()

    labels.img_hw = (rh, rw)
    return im, labels


def augment_PIL(imgs: List[PIL.Image.Image], labels: List[ImageObjects],
                aug_setting: dict):
    '''
    Perform random augmentation for a list of PIL images.

    Args:
        imgs:
        labels:
        aug_Setting:

    Return:
        imgs:
        labels:
    '''
    assert len(imgs) == len(labels)
    num = len(imgs)
    if torch.rand(1).item() > 0.5:
        low, high = aug_setting.get('brightness', [0.6, 1.4])
        _val = uniform(low, high)
        for i in range(num):
            imgs[i] = tvf.adjust_brightness(imgs[i], _val)
    if torch.rand(1).item() > 0.5:
        low, high = aug_setting.get('contrast', [0.5, 1.5])
        _val = uniform(low, high)
        for i in range(num):
            imgs[i] = tvf.adjust_contrast(imgs[i], _val)
    if torch.rand(1).item() > 0.5:
        low, high = aug_setting.get('hue', [-0.1, 0.1])
        _val = uniform(low, high)
        for i in range(num):
            imgs[i] = tvf.adjust_hue(imgs[i], _val)
    if torch.rand(1).item() > 0.5:
        low, high = aug_setting.get('saturation', [0, 2])
        _val = uniform(low, high)
        for i in range(num):
            imgs[i] = tvf.adjust_saturation(imgs[i], _val)
    # if torch.rand(1).item() > 0.5:
    #     img = tvf.adjust_gamma(img, uniform(0.5, 3))
    # horizontal flip
    if aug_setting['horizontal_flip'] and torch.rand(1).item() > 0.5:
        for i in range(num):
            imgs[i], labels[i] = hflip(imgs[i], labels[i])
    # vertical flip
    if aug_setting['vertical_flip'] and torch.rand(1).item() > 0.5:
        for i in range(num):
            imgs[i], labels[i] = vflip(imgs[i], labels[i])
    if aug_setting['rotation']:
        # random rotation
        rand_deg = torch.rand(1).item() * 360
        expand = aug_setting['rotation_expand']
        for i in range(num):
            imgs[i], labels[i] = rotate(imgs[i], rand_deg, labels[i], expand=expand)
    return imgs, labels


def random_place(img: PIL.Image.Image, labels: ImageObjects,
                 background: PIL.Image.Image, dt=0.1):
    '''
    Random place the objects into the background image.

    labels will be unchanged.

    Args:
        img: current image
        labels: labels for img
        background: background image generated by inpainting methods
        dt: time interval between frames. e.g., dt=0.1 means 10 fps
    '''
    assert labels._bb_format == 'cxcywhd'
    im = np.array(img).astype('float32') / 255
    bg = np.array(background).astype('float32') / 255
    plt.figure(figsize=(8,8)); plt.imshow(bg)
    
    npmasks = labels.masks.numpy()
    for mask in npmasks:
        # randomly disappear
        if torch.rand(1).item() < 0.01:
            continue
        # mask: torch.BoolTensor
        # npmask = mask.numpy()
        masked = im * np.expand_dims(mask, axis=2)
        # select the smallest window that contains this mask
        idx0, idx1 = np.nonzero(mask)
        # (ymin, xmin), _ = torch.min(idxs, dim=0)
        # (ymax, xmax), _ = torch.max(idxs, dim=0)
        ymin, ymax = np.min(idx0), np.max(idx0)
        xmin, xmax = np.min(idx1), np.max(idx1)
        window = masked[ymin:ymax+1, xmin:xmax+1, :]
        mask_win = mask[ymin:ymax+1, xmin:xmax+1]
        # plt.figure(figsize=(8,8)); plt.imshow(mask_win, cmap='gray'); plt.show()
        # location and width
        wincx, wincy = (xmin + xmax) / 2, (ymin + ymax) / 2
        winh, winw = window.shape[:2]
        # plt.figure(); plt.imshow(window); plt.show()
        # convert to PIL.Image
        # window = PIL.Image.fromarray(window)
        # windowr = window.rotate(theta, expand=True)
        wh_max = max(xmax-xmin, ymax-ymin)
        # randomly resize
        new_h, new_w = int(winh*normal(1,1*dt)), int(winw*normal(1,1*dt))
        window = cv2.resize(window, (new_w, new_h))
        mask_win = cv2.resize(mask_win.astype('float32'), (new_w, new_h))
        # rotate by a random angle
        da = normal(0, 80) * dt
        win_rot = scipy.ndimage.rotate(window, angle=da, reshape=True, prefilter=False)
        mask_win = scipy.ndimage.rotate(mask_win, angle=da, reshape=True, prefilter=False)
        mask_win = (mask_win > 0.5)
        new_h, new_w = win_rot.shape[:2]
        # plt.figure(); plt.imshow(window)
        # plt.figure(); plt.imshow(win_rot); plt.show()
        # randomly move
        new_cx = int(wincx + normal(0, wh_max*dt))
        new_cy = int(wincy + normal(0, wh_max*dt))
        # place it on the background
        _x1, _y1 = max(new_cx - new_w//2, 0), max(new_cy - new_h//2, 0)
        _x2, _y2 = max(_x1 + new_w,0), max(_y1 + new_h,0)
        bg_window = bg[_y1:_y2, _x1:_x2, :]
        _h, _w = bg_window.shape[:2]
        if _h == 0 or _w == 0:
            continue
        # edge cases
        win_rot, mask_win = _random_crop(win_rot, mask_win, _h, _w)
        # plt.figure(); plt.imshow(bg_window)
        # plt.figure(); plt.imshow(win_rot); plt.show()
        # mask_rot = win_rot.sum(axis=2) > 0.5
        bg_window[mask_win] = win_rot[mask_win]
        bg[_y1:_y2, _x1:_x2, :] = bg_window
    plt.figure(figsize=(8,8)); plt.imshow(bg)
    plt.figure(figsize=(8,8)); plt.imshow(im); plt.show()
    raise NotImplementedError()


def _random_crop(im, mask, win_h, win_w):
    assert 0 < win_h <= im.shape[0] and 0 < win_w <= im.shape[1]
    _y1 = random.randint(0, im.shape[0] - win_h)
    _x1 = random.randint(0, im.shape[1] - win_w)
    im = im[_y1:_y1+win_h, _x1:_x1+win_w, :]
    mask = mask[_y1:_y1+win_h, _x1:_x1+win_w] if mask is not None else None
    return im, mask


def hflip(image: PIL.Image.Image, labels: ImageObjects):
    '''
    left-right flip
    '''
    image = tvf.hflip(image)

    assert labels._bb_format in {'cxcywh', 'cxcywhd'}
    labels.bboxes[:,0] = image.width - labels.bboxes[:,0]
    if labels._bb_format == 'cxcywhd':
        labels.bboxes[:,4] = -labels.bboxes[:,4]
    if labels.masks is not None:
        assert labels.masks.dim() == 3
        labels.masks = torch.from_numpy(np.flip(labels.masks.numpy(), axis=2).copy())
    return image, labels


def vflip(image: PIL.Image.Image, labels: ImageObjects):
    '''
    up-down flip
    '''
    image = tvf.vflip(image)

    assert labels._bb_format == 'cxcywhd'
    labels.bboxes[:,1] = image.height - labels.bboxes[:,1] # x,y,w,h,angle
    labels.bboxes[:,4] = -labels.bboxes[:,4]
    if labels.masks is not None:
        assert labels.masks.dim() == 3
        labels.masks = torch.from_numpy(np.flip(labels.masks.numpy(), axis=1).copy())
    return image, labels


def rotate(image: PIL.Image.Image, degrees, labels: ImageObjects, expand=False):
    '''
    Rotate the image and the labels by degrees counter-clockwise

    Args:
        expand: if True, the image border will be expanded after rotation
    '''
    if labels.masks is not None:
        raise NotImplementedError()
    img_w, img_h = image.width, image.height
    image = tvf.rotate(image, angle=-degrees, expand=expand)
    new_w, new_h = image.width, image.height
    # image coordinate to cartesian coordinate
    x = labels.bboxes[:,0] - 0.5*img_w
    y = -(labels.bboxes[:,1] - 0.5*img_h)
    # cartesian to polar
    r = (x.pow(2) + y.pow(2)).sqrt()

    theta = torch.empty_like(r)
    theta[x>=0] = torch.atan(y[x>=0]/x[x>=0])
    theta[x<0] = torch.atan(y[x<0]/x[x<0]) + np.pi
    theta[torch.isnan(theta)] = 0
    # modify theta
    theta -= (degrees*np.pi/180)
    # polar to cartesian
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    labels.bboxes[:,0] = x + 0.5*new_w
    labels.bboxes[:,1] = -y + 0.5*new_h
    labels.bboxes[:,4] += degrees
    labels.bboxes[:,4] = torch.remainder(labels.bboxes[:,4], 180)
    labels.bboxes[:,4][labels.bboxes[:,4]>=90] -= 180
    labels.img_hw = (new_h, new_w)

    return image, labels


def add_gaussian(imgs, max_var=0.1):
    '''
    imgs: tensor, (batch),C,H,W
    max_var: variance is uniformly ditributed between 0~max_var
    '''
    var = torch.rand(1) * max_var
    imgs = imgs + torch.randn_like(imgs) * var

    return imgs


def add_saltpepper(imgs: torch.FloatTensor, max_p=0.06):
    '''
    Add salt & pepper noise to the image in-place

    Args:
        imgs: tensor, (batch),C,H,W
        p: probibility to add salt and pepper
    '''
    c,h,w = imgs.shape[-3:]

    p = torch.rand(1) * max_p
    total = int(c*h*w * p)

    idxC = torch.randint(0,c,size=(total,))
    idxH = torch.randint(0,h,size=(total,))
    idxW = torch.randint(0,w,size=(total,))
    value = torch.randint(0,2,size=(total,),dtype=torch.float)

    imgs[...,idxC,idxH,idxW] = value
    return imgs


def random_avg_filter(img, kernel_sizes=[3]):
    assert img.dim() == 3
    img = img.unsqueeze(0)
    ks = random.choice(kernel_sizes)
    pad_size = ks // 2
    img = tnf.avg_pool2d(img, kernel_size=ks, stride=1, padding=pad_size)
    return img.squeeze(0)


def max_filter(img):
    assert img.dim() == 3
    img = img.unsqueeze(0)
    img = tnf.max_pool2d(img, kernel_size=3, stride=1, padding=1)
    return img.squeeze(0)


def get_gaussian_kernels(kernel_sizes=[3,5]):
    gaussian_kernels = []
    for ks in kernel_sizes:
        delta = np.zeros((ks,ks))
        delta[ks//2,ks//2] = 1
        kernel = scipy.ndimage.gaussian_filter(delta, sigma=3)
        kernel = torch.from_numpy(kernel).float().view(1,1,ks,ks)
        gaussian_kernels.append(kernel)
    return gaussian_kernels

gaussian_kernels = get_gaussian_kernels(kernel_sizes=[3])
def random_gaussian_filter(img):
    assert img.dim() == 3
    img = img.unsqueeze(1)
    kernel = random.choice(gaussian_kernels)
    assert torch.isclose(kernel.sum(), torch.Tensor([1]))
    pad_size = kernel.shape[2] // 2
    img = tnf.conv2d(img, weight=kernel, stride=1, padding=pad_size)
    return img.squeeze(1)


def uniform(a, b):
    '''
    Sample a real number between a and b according to a uniform distribution
    '''
    assert a <= b
    return a + torch.rand(1).item() * (b-a)


def normal(mean, std):
    return mean + torch.randn(1).item() * std


if __name__ == "__main__":
    from PIL import Image
    img_path = 'C:/Projects/MW18Mar/train_no19/Mar10_000291.jpg'
    img = Image.open(img_path)
    img.show()

    new_img = tvf.rotate(img, -45)
    new_img.show()