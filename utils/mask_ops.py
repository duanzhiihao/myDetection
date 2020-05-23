import numpy as np
import torch
import pycocotools.mask as _maskUtils


def segm2rle(segmentation: list, img_h: int, img_w: int):
    '''
    Convert polygon, bbox, and uncompressed RLE to encoded RLE mask.

    Args:
        segmentation: list from COCO json
        img_h, img_h: image height and width
    '''
    if isinstance(segmentation, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = _maskUtils.frPyObjects(segmentation, img_h, img_w)
        rle = _maskUtils.merge(rles)
    elif isinstance(segmentation['counts'], list):
        # uncompressed RLE
        rle = _maskUtils.frPyObjects(segmentation, img_h, img_w)
    else:
        # already RLE
        rle = segmentation
    return rle


def rle2mask(rles) -> torch.BoolTensor:
    '''
    Decode binary masks encoded by Run Length Encoding (RLE).

    Args:
        rles: rle or list of rles
    
    Return:
        masks: torch.BoolTensor, shape[len(rles), image h, image w]
    '''
    masks = _maskUtils.decode(rles)
    assert masks.dtype == np.uint8 and masks.ndim == 3
    masks = torch.from_numpy(masks).bool().permute(2, 0, 1)
    return masks

