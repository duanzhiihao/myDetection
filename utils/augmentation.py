import random
import numpy as np
import scipy.ndimage
import torch
import torchvision.transforms.functional as tvf
import torch.nn.functional as tnf

from .structures import ImageObjects


def hflip(image, labels):
    '''
    left-right flip

    Args:
        image: PIL.Image
    '''
    assert isinstance(labels, ImageObjects)
    image = tvf.hflip(image)
    assert labels._bb_format in {'cxcywh', 'cxcywhd'}
    labels.bboxes[:,0] = image.width - labels.bboxes[:,0]
    if labels._bb_format == 'cxcywhd':
        labels.bboxes[:,4] = -labels.bboxes[:,4]
    return image, labels


def vflip(image, labels):
    '''
    up-down flip

    Args:
        image: PIL.Image
    '''
    assert isinstance(labels, ImageObjects)
    assert labels._bb_format == 'cxcywhd'
    image = tvf.vflip(image)
    labels.bboxes[:,1] = image.height - labels.bboxes[:,1] # x,y,w,h,(angle)
    labels.bboxes[:,4] = -labels.bboxes[:,4]
    return image, labels


def rotate(image, degrees, labels, expand=False):
    '''
    Args:
        image: PIL.Image
    '''
    assert isinstance(labels, ImageObjects)
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


def add_saltpepper(imgs, max_p=0.06):
    '''
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


if __name__ == "__main__":
    from PIL import Image
    img_path = 'C:/Projects/MW18Mar/train_no19/Mar10_000291.jpg'
    img = Image.open(img_path)
    img.show()

    new_img = tvf.rotate(img, -45)
    new_img.show()