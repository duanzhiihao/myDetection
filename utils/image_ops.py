import numpy as np
import PIL.Image
import random
import torch
import torchvision.transforms.functional as tvf

from .structures import ImageObjects


def imread_pil(img_path):
    img = PIL.Image.open(img_path)
    if img.mode == 'L':
        # print(f'Warning: image {img_path} is grayscale')
        img = np.array(img)
        img = np.repeat(np.expand_dims(img,2), 3, axis=2)
        img = PIL.Image.fromarray(img)
    return img


def pad_to_divisible(img: PIL.Image.Image, denom: int) -> PIL.Image.Image:
    '''
    Zero-pad at the right and bottom of the image such that width and height
    are divisible by `denom`
    
    Args:
        img: PIL.Image.Image
        denom: int, the desired denominator
    '''
    img_h, img_w = img.height, img.width
    pad_bottom = int(np.ceil(img_h/denom) * denom) - img_h
    pad_right = int(np.ceil(img_w/denom) * denom) - img_w
    assert 0 <= pad_bottom < denom and 0 <= pad_right < denom
    pil_img = tvf.pad(img, padding=(0,0,pad_right,pad_bottom), fill=0)
    return pil_img


def rect_to_square(img: PIL.Image.Image, labels: ImageObjects,
                   target_size:int, pad_value:int=0, aug:bool=False,
                   resize_step=128):
    '''
    Args:
        img: PIL.Image
        labels: ImageObjects
        target_size: int, e.g. 608
        pad_value: int
        aug: bool
    '''
    assert isinstance(img, PIL.Image.Image) and img.mode == 'RGB'
    ori_h, ori_w = img.height, img.width

    # resize to target input size (usually smaller)
    resize_scale = target_size / max(ori_w,ori_h)
    if aug:
        low_ = (target_size - resize_step) / target_size
        resize_scale = resize_scale * (low_ + np.random.rand()*(1-low_))
    assert resize_scale > 0
    nopad_w, nopad_h = int(ori_w*resize_scale), int(ori_h*resize_scale)
    img = tvf.resize(img, (nopad_h,nopad_w))

    # pad to square
    if aug:
        # random placing
        left = random.randint(0, target_size - nopad_w)
        top = random.randint(0, target_size - nopad_h)
    else:
        left = (target_size - nopad_w) // 2
        top = (target_size - nopad_h) // 2
    right = target_size - nopad_w - left
    bottom = target_size - nopad_h - top

    fill_value = tuple([random.randint(0,255) for _ in range(3)]) if aug else 0
    img = tvf.pad(img, padding=(left,top,right,bottom), fill=fill_value)
    # record the padding info
    img_tl = (left, top) # start of the true image
    img_wh = (nopad_w, nopad_h)

    # modify labels
    if labels is not None:
        assert isinstance(labels, ImageObjects)
        assert labels._bb_format in {'cxcywh', 'cxcywhd'}
        labels.bboxes[:,:4] *= resize_scale
        labels.bboxes[:,0] += left
        labels.bboxes[:,1] += top
        if labels.img_hw is not None:
            assert labels.img_hw == (ori_h, ori_w)
        labels.img_hw = (target_size, target_size)
    
    pad_info = (ori_w, ori_h) + img_tl + img_wh
    assert isinstance(pad_info, tuple)
    return img, labels, pad_info


def format_tensor_img(t_img: torch.tensor, code: str):
    '''
    Args:
        code: str
    '''
    assert torch.is_tensor(t_img) and t_img.dim() == 3 and t_img.shape[0] == 3
    assert 0 < t_img.mean() < 1
    if code == 'RGB_1':
        pass
    elif code == 'RGB_1_norm':
        means = [0.485,0.456,0.406]
        stds = [0.229,0.224,0.225]
        t_img = tvf.normalize(t_img, means, stds)
    elif code == 'BGR_255_norm':
        # to BGR, to 255
        t_img = t_img[[2,1,0],:,:] * 255
        # normalization
        t_img = tvf.normalize(t_img, [102.9801,115.9465,122.7717], [1,1,1])
    else:
        raise NotImplementedError()
    return t_img


def tensor_img_to_pil(t_img: torch.tensor, code: str):
    assert torch.is_tensor(t_img) and t_img.dim() == 3 and t_img.shape[0] == 3
    # assert 0 < t_img.mean() < 1
    if code == 'RGB_1':
        pass
    elif code == 'RGB_1_norm':
        means = [0.485,0.456,0.406]
        stds = [0.229,0.224,0.225]
        for channel, m, sd in zip(t_img, means, stds):
            channel.mul_(sd).add_(m)
    elif code == 'BGR_255_norm':
        raise NotImplementedError()
    else:
        raise NotImplementedError()
    return tvf.to_pil_image(t_img)
