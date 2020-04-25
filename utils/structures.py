import functools
import torch
import torchvision

from .bbox_ops import nms_rotbb
from .visualization import draw_bboxes_on_np
from .constants import COCO_CATEGORY_LIST

class ImageObjects():
    '''
    A group of image bounding boxes

    Args:
        bboxes: 2-d tensor, torch.float32
        cats: 1-d tensor, torch.int64, categories
        scores (optional): 1-d tensor, torch.float32, scores
        bb_format (optional): str, can be:
                'cxcywh': (cx,cy,w,h)
                'x1y1x2y2': (x1,y1,x2,y2)
                'cxcywhd': (cx,cy,w,h,degree)
                'cxcywhr': (cx,cy,w,h,radian)
        img_size: int or (height, width)
    '''
    def __init__(self, bboxes, cats, scores=None,
                       bb_format='cxcywh', img_size=None):
        # format check
        assert bboxes.dim() == 2 and bboxes.shape[0] == cats.shape[0]
        assert cats.dtype == torch.int64, 'Incorrect data type of categories'
        if bb_format == 'cxcywh':
            assert bboxes.shape[1] == 4
        elif bb_format == 'cxcywhd':
            assert bboxes.shape[1] == 5
        else:
            raise NotImplementedError()
        # if cats is None:
        #     cats = torch.zeros(bboxes.shape[:-1], dtype=torch.int64)

        self.bboxes = bboxes
        self.cats = cats
        self.scores = scores
        self._bb_format = bb_format
        self.img_size = img_size
    
    def __getitem__(self, idx):
        b = self.bboxes[idx,:]
        c = self.cats[idx]
        s = self.scores[idx] if self.scores is not None else None
        return ImageObjects(b, c, s, self._bb_format)
    
    def __len__(self):
        obj_num = self.bboxes.shape[0]
        return obj_num
    
    def cpu_(self):
        self.bboxes = self.bboxes.cpu()
        self.cats = self.cats.cpu()
        self.scores = self.scores.cpu() if self.scores is not None else None

    def nms(self, nms_thres=0.45):
        return ImageObjects.non_max_suppression(self, nms_thres)

    @staticmethod
    def non_max_suppression(dts, nms_thres: float):
        '''
        Non-maximum suppression for bounding boxes
        
        Args:
            dts: ImageObjects
            nms_thres: float
        '''
        assert isinstance(dts, ImageObjects)
        assert dts.scores is not None
        if dts.bboxes.shape[0] == 0:
            return dts

        if dts._bb_format == 'cxcywh':
            # converting to x1y1x2y2
            bbs = dts.bboxes.clone()
            bbs[:,0] = dts.bboxes[:,0] - dts.bboxes[:,2]/2
            bbs[:,1] = dts.bboxes[:,1] - dts.bboxes[:,3]/2
            bbs[:,2] = dts.bboxes[:,0] + dts.bboxes[:,2]/2
            bbs[:,3] = dts.bboxes[:,1] + dts.bboxes[:,3]/2
            single_cls_nms_func = torchvision.ops.nms
        elif dts._bb_format == 'x1y1x2y2':
            bbs = dts.bboxes.clone()
            single_cls_nms_func = torchvision.ops.nms
        elif dts._bb_format == 'cxcywhd':
            bbs = dts.bboxes.clone()
            def single_cls_nms_func(boxes, socres, nms_thres):
                img_size = dts.img_size or 1024
                return nms_rotbb(boxes, scores, nms_thres, bb_format=dts._bb_format,
                                 img_size=img_size, majority=None)
        else:
            raise NotImplementedError()

        scores, categories = dts.scores, dts.cats
        out_bbs, out_scores, out_cats = [], [], []
        unique_labels = categories.unique()
        for cat_idx in unique_labels:
            cls_mask = (categories==cat_idx)

            keep_indices = single_cls_nms_func(bbs[cls_mask,:], scores[cls_mask],
                                               nms_thres)

            out_bbs.append(dts.bboxes[cls_mask,:][keep_indices,:])
            out_scores.append(scores[cls_mask][keep_indices])
            out_cats.append(categories[cls_mask][keep_indices])
        
        out_bbs = torch.cat(out_bbs, dim=0)
        out_scores = torch.cat(out_scores, dim=0)
        out_cats = torch.cat(out_cats, dim=0)
        return ImageObjects(out_bbs, out_cats, out_scores, dts._bb_format)
    
    # def _bbox_cvt_format(self, target_format: str):
    #     '''
    #     Args:
    #         target_format: str
    #     '''
    #     if target_format == 'x1y1x2y2':
    #         assert self.bboxes.dim() == 2 and self.bboxes.shape[1] == 4
    #         if self._bb_format == 'x1y1x2y2':
    #             pass
    #         elif self._bb_format == 'cxcywh':
    #             bbs = torch.empty_like(self.bboxes)
    #             bbs[:,0] = self.bboxes[:,0] - self.bboxes[:,2]/2
    #             bbs[:,1] = self.bboxes[:,1] - self.bboxes[:,3]/2
    #             bbs[:,2] = self.bboxes[:,0] + self.bboxes[:,2]/2
    #             bbs[:,3] = self.bboxes[:,1] + self.bboxes[:,3]/2
    #             self.bboxes = bbs
    #         else: raise NotImplementedError()
    #     elif target_format == 'cxcywh':
    #         assert self.bboxes.dim() == 2 and self.bboxes.shape[1] == 4
    #         if self._bb_format == 'cxcywh':
    #             pass
    #         elif self._bb_format == 'x1y1x2y2':
    #             bbs = torch.empty_like(self.bboxes)
    #             bbs[:,0] = (self.bboxes[:,0] + self.bboxes[:,2]) / 2
    #             bbs[:,1] = (self.bboxes[:,1] + self.bboxes[:,3]) / 2
    #             bbs[:,2] = self.bboxes[:,2] - self.bboxes[:,0]
    #             bbs[:,3] = self.bboxes[:,3] - self.bboxes[:,1]
    #             self.bboxes = bbs
    #         else: raise NotImplementedError()
    #     else:
    #         raise NotImplementedError()
    #     self._bb_format = target_format

    def bboxes_to_original_(self, pad_info):
        '''
        Recover the bbox from the padded square image to in the original image.

        Args:
            pad_info: (ori w, ori h, tl x, tl y, imw, imh)
        '''
        assert len(pad_info) == 6
        ori_w, ori_h, tl_x, tl_y, imw, imh = pad_info
        self.bboxes[:,0] = (self.bboxes[:,0] - tl_x) / imw * ori_w
        self.bboxes[:,1] = (self.bboxes[:,1] - tl_y) / imh * ori_h
        self.bboxes[:,2] = self.bboxes[:,2] / imw * ori_w
        self.bboxes[:,3] = self.bboxes[:,3] / imh * ori_h
    
    def sanity_check(self):
        # Check dimension and length
        assert self.bboxes.dim() == 2
        bbox_num = self.bboxes.shape[0]
        assert self.cats.shape[0] == bbox_num
        if self.scores is not None:
            assert self.scores.shape[0] == bbox_num
        # Check bbox format
        if self._bb_format == 'cxcywh':
            assert self.bboxes.shape[1] == 4
        elif self._bb_format == 'cxcywhd':
            assert self.bboxes.shape[1] == 5
        else:
            raise NotImplementedError()
        # Check dtype
        assert self.cats.dtype == torch.int64, 'Incorrect data type of categories'
    
    def draw_on_np(self, im, class_map='COCO', **kwargs):
        assert self.bboxes.dim() == 2
        draw_bboxes_on_np(im, self, class_map=class_map, **kwargs)

    def to_coco_json(self, img_id):
        assert self.bboxes.dim() == 2
        assert self.bboxes.shape[0] == self.cats.shape[0] == self.scores.shape[0]
        list_json = []
        for bb, c, s in zip(self.bboxes, self.cats, self.scores):
            if self._bb_format == 'cxcywh':
                cx,cy,w,h = [float(t) for t in bb]
                bbox = [cx-w/2, cy-h/2, w, h]
            else: # TODO
                raise NotImplementedError()
            cat_id = COCO_CATEGORY_LIST[int(c)]['id']
            dt_dict = {'image_id': img_id, 'category_id': cat_id,
                       'bbox': bbox, 'score': float(s)}
            list_json.append(dt_dict)
        return list_json
