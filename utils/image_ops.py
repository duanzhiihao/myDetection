import random
import numpy as np
import PIL.Image
import cv2
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


def rect_to_square(image, labels, target_size:int, aug:bool=False,
                   resize_step:int=128):
    '''
    Resize and pad the input image to square. \\
    If aug, the input image will be randomly resized such that the longer side is \
        between [target_size-resize_step, target_size].\\
    Otherwise, the input is resied such that longer side = target_size.

    Args:
        image: PIL.Image or list of PIL.Image
        labels: ImageObjects or list of ImageObjects
        target_size: int, the width/height of the output square
        aug: bool, if Ture, perform random resizing and placing augmentation
        resize_step: int, the input image is re
    '''
    if isinstance(image, PIL.Image.Image):
        imgs, labels = [image], [labels]
    else:
        imgs = image
    if labels is not None:
        assert len(imgs) == len(labels)
    pad_info = []
    aug_resize = torch.rand(1).item()
    fill_value = tuple([random.randint(0,255) for _ in range(3)]) if aug else 0
    for i in range(len(imgs)):
        _img = imgs[i]
        assert isinstance(_img, PIL.Image.Image) and _img.mode == 'RGB'
    
        ori_h, ori_w = _img.height, _img.width
        # resize to target input size (usually smaller)
        resize_scale = target_size / max(ori_w,ori_h)
        if aug:
            low_ = (target_size - resize_step) / target_size
            resize_scale = resize_scale * (low_ + aug_resize*(1-low_))
        assert resize_scale > 0
        nopad_w, nopad_h = int(ori_w*resize_scale), int(ori_h*resize_scale)
        _img = tvf.resize(_img, (nopad_h,nopad_w))

        # pad to square
        if aug and isinstance(image, PIL.Image.Image):
            # random placing if enable aug. and input is a single image
            left = random.randint(0, target_size - nopad_w)
            top = random.randint(0, target_size - nopad_h)
        else:
            left = (target_size - nopad_w) // 2
            top = (target_size - nopad_h) // 2
        right = target_size - nopad_w - left
        bottom = target_size - nopad_h - top

        _img = tvf.pad(_img, padding=(left,top,right,bottom), fill=fill_value)
        imgs[i] = _img

        # record the padding info
        img_tl = (left, top) # start of the true image
        img_wh = (nopad_w, nopad_h)
        _info = (ori_w, ori_h) + img_tl + img_wh
        pad_info.append(_info)

        # modify labels
        _lab = labels[i]
        if _lab is not None:
            assert isinstance(_lab, ImageObjects)
            assert _lab._bb_format in {'cxcywh', 'cxcywhd'}
            _lab.bboxes[:,:4] *= resize_scale
            _lab.bboxes[:,0] += left
            _lab.bboxes[:,1] += top
            if _lab.img_hw is not None:
                assert _lab.img_hw == (ori_h, ori_w)
            _lab.img_hw = (target_size, target_size)
        
    if isinstance(image, PIL.Image.Image):
        image, labels, pad_info = imgs[0], labels[0], pad_info[0]
    else:
        image = imgs
    return image, labels, pad_info


def format_tensor_img(t_img: torch.FloatTensor, code: str) -> torch.FloatTensor:
    '''
    Transform the tensor image to a specified format.

    Args:
        t_img: tensor image. must be torch.FloatTensor between 0-1
        code: str
    '''
    assert isinstance(t_img, torch.FloatTensor) and 0 <= t_img.mean() <= 1
    assert t_img.dim() == 3 and t_img.shape[0] == 3
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


def img_tensor_to_np(t_img: torch.FloatTensor, encoding: str, out_format: str):
    '''
    Convert a tensor image to numpy image. 
    This is sort of the inverse operation of format_tensor_img(). \\
    NOTE: this function is not optimized for speed

    Args:
        t_img: tensor image
        encoding: how tensor image is transformed.
                  Available: 'RGB_1', 'RGB_1_norm', 'BGR_255_norm'
        out_format: 'RGB_1', 'BGR_1'
    '''
    assert torch.is_tensor(t_img) and t_img.dim() == 3 and t_img.shape[0] == 3
    assert encoding in {'RGB_1', 'RGB_1_norm', 'BGR_255_norm'}
    assert out_format in {'RGB_1', 'BGR_1', 'BGR_uint8'}

    t_img = t_img.clone()
    # 0. convert everthing to RGB_1
    if encoding == 'RGB_1':
        pass
    elif encoding == 'RGB_1_norm':
        means = [0.485,0.456,0.406]
        stds = [0.229,0.224,0.225]
        for channel, m, sd in zip(t_img, means, stds):
            channel.mul_(sd).add_(m)
    elif encoding == 'BGR_255_norm':
        raise NotImplementedError()
    else:
        raise NotImplementedError()
    im = t_img.permute(1, 2, 0).numpy()
    # 1. convert RGB_1 to output format
    if out_format == 'RGB_1':
        pass
    elif out_format == 'BGR_1':
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    elif out_format == 'BGR_uint8':
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = (im * 255).astype('uint8')
    return im


def extract_mask(im: np.ndarray, mask: np.ndarray, cxcywhd):
    '''
    extract a bounding box from an image according to a binary mask and rotbbox
    '''
    raise NotImplementedError()
    assert mask.dtype == np.bool and mask.ndim == 2 and mask.shape == im.shape[:2]
    assert len(cxcywhd) == 5

    if isinstance(im, PIL.Image.Image):
        raise NotImplementedError()
    assert isinstance(im, np.ndarray)

    # idx0, idx1 = np.nonzero(mask)
    # ymin, xmin = np.min(idx0), np.min(idx1)
    # ymax, xmax = np.max(idx0), np.max(idx1)
    # center = [(xmin + xmax) / 2, (ymin + ymax) / 2]
    # w = masked[ymin:ymax+1, xmin:xmax+1]

    # roate the whole image

