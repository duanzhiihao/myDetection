import random
import numpy as np
import cv2
import PIL.Image
import matplotlib.pyplot as plt
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


def resize_pil(img: PIL.Image.Image, img_size: int, shorter=True):
    '''
    Resize the PIL image such that the shorter/longer side = img_size
    '''
    # assert isinstance(img_size, int)
    if shorter:
        pass
    else:
        imh, imw = img.height, img.width
        factor = img_size / max(imh, imw)
        th, tw = round(imh*factor), round(imw*factor)
        img_size = (th, tw)
    img = tvf.resize(img, size=img_size)
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


def rect_to_square(image, labels, target_size:int, aug=False, resize_step=128,
                   _aug_rsz=1, _fill_val=None, _rand_plc=True):
    '''
    Resize and pad the input image to square.

    If aug == True,
        1. The input image will be randomly resized such that the longer side is
           between [target_size-resize_step, target_size].
        2. The input image will be randomly placed on a colored background.
    Otherwise,
        1. the input will be resied such that longer side = target_size.
        2. The input image will be zero-padded to square.

    Args:
        image:       PIL.Image or list of PIL.Image
        labels:      ImageObjects or list of ImageObjects
        target_size: int, the width/height of the output square
        aug:         bool, if Ture, perform random resizing and placing augmentation
        resize_step: int
    '''
    assert isinstance(image, PIL.Image.Image) and image.mode == 'RGB'
    if labels is not None:
        assert isinstance(labels, ImageObjects)

    # augmentation settings
    _aug_rsz = _aug_rsz or torch.rand(1).item()
    if aug:
        _fill_val = _fill_val or tuple([random.randint(0,255) for _ in range(3)])
    else:
        _fill_val = 0

    ori_h, ori_w = image.height, image.width
    # resize to target input size
    resize_scale = target_size / max(ori_w,ori_h)
    if aug:
        low_ = (target_size - resize_step) / target_size
        resize_scale = resize_scale * (low_ + _aug_rsz*(1-low_))
    assert resize_scale > 0
    resized_w, resized_h = int(ori_w*resize_scale), int(ori_h*resize_scale)
    image = tvf.resize(image, (resized_h,resized_w))

    # random placing parematers
    if aug and _rand_plc:
        left = random.randint(0, target_size - resized_w)
        top = random.randint(0, target_size - resized_h)
    else:
        left = (target_size - resized_w) // 2
        top = (target_size - resized_h) // 2
    right = target_size - resized_w - left
    bottom = target_size - resized_h - top
    # pad to square
    image = tvf.pad(image, padding=(left,top,right,bottom), fill=_fill_val)

    # record the padding info
    pad_info = (ori_w, ori_h, left, top, resized_w, resized_h)

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
        if labels.masks is not None:
            old_masks = labels.masks
            assert old_masks.shape[1:] == (ori_h, ori_w)
            new_masks = torch.zeros(old_masks.shape[0], target_size, target_size,
                                    dtype=torch.bool)
            for i in range(old_masks.shape[0]):
                m = PIL.Image.fromarray(old_masks[i].numpy())
                # plt.figure(figsize=(8,8)); plt.imshow(m, cmap='gray')
                m = tvf.resize(m, (resized_h,resized_w))
                m = tvf.pad(m, padding=(left,top,right,bottom), fill=0)
                # plt.figure(figsize=(8,8)); plt.imshow(m, cmap='gray')
                # plt.show()
                new_masks[i] = torch.from_numpy(np.array(m, dtype=np.bool))
            labels.masks = new_masks
        labels.sanity_check()

    return image, labels, pad_info


def seq_rect_to_square(images, labels, target_size:int, aug:bool=False,
                       resize_step:int=128):
    '''
    Resize and pad a sequence of image to square.

    See rect_to_square() for detailed docs.
    '''
    assert isinstance(images, list)
    if labels is not None:
        assert isinstance(labels, list) and len(images) == len(labels)
    _aug_rsz = torch.rand(1).item() if aug else 1
    _fill_val = tuple([random.randint(0,255) for _ in range(3)]) if aug else 0
    new_images = []
    new_labels = []
    pad_infos  = []
    for i in range(len(images)):
        img, lbl, info = rect_to_square(images[i], labels[i], target_size=target_size,
                            aug=aug, resize_step=resize_step,
                            _aug_rsz=_aug_rsz, _fill_val=_fill_val, _rand_plc=False)
        new_images.append(img)
        new_labels.append(lbl)
        pad_infos.append(info)
    return new_images, new_labels, pad_infos


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
        stds  = [0.229,0.224,0.225]
        t_img = tvf.normalize(t_img, means, stds)
    elif code == 'BGR_255_norm':
        # to BGR, to 255
        t_img = t_img[[2,1,0],:,:] * 255
        # normalization
        t_img = tvf.normalize(t_img, [102.9801,115.9465,122.7717], [1,1,1])
    else:
        raise NotImplementedError()
    return t_img


def tensor_to_np(t_img: torch.FloatTensor, encoding: str, out_format: str):
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
    assert out_format in {'RGB_1', 'BGR_1', 'BGR_uint8', 'RGB_uint8'}

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
    elif out_format == 'RGB_uint8':
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = (im * 255).astype('uint8')
    elif out_format == 'BGR_uint8':
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = (im * 255).astype('uint8')
    else:
        raise NotImplementedError()
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

