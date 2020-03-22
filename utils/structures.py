import torch
import torchvision

from .visualization import draw_cocobb_on_np
from .utils import COCO_CATEGORY_LIST

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
    '''
    def __init__(self, bboxes, cats, scores, bb_format='cxcywh'):
        # format check
        assert bboxes.dim() == 2 and bboxes.shape[0] == cats.shape[0]
        if bb_format == 'cxcywh':
            assert bboxes.shape[-1] == 4
        else:
            raise NotImplementedError()
        # if cats is None:
        #     cats = torch.zeros(bboxes.shape[:-1], dtype=torch.int64)

        self.bboxes = bboxes
        self.cats = cats
        self.scores = scores
        self._bb_format = bb_format
    
    def __getitem__(self, idx):
        b = self.bboxes[idx,:]
        c = self.cats[idx]
        s = self.scores[idx]
        return ImageObjects(b, c, s, self._bb_format)
    
    def __len__(self):
        return self.bboxes.shape[0]
    
    def cpu_(self):
        self.bboxes = self.bboxes.cpu()
        self.cats = self.cats.cpu()
        self.scores = self.scores.cpu()

    def nms(self, nms_thres=0.45):
        return ImageObjects.nms_(self, nms_thres)

    @staticmethod
    def nms_(dts, nms_thres=0.45):
        '''
        Non-maximum suppression for bounding boxes
        
        Args:
            majority (optional): int, a BB is suppresssed if the number of votes \
                less than majority. Typically used with test-time augmentation.
        '''
        assert isinstance(dts, ImageObjects)
        if dts.bboxes.shape[0] == 0:
            return dts

        if dts._bb_format == 'cxcywh':
            # converting to x1y1x2y2
            bbs = dts.bboxes.clone()
            bbs[:,0] = dts.bboxes[:,0] - dts.bboxes[:,2]/2
            bbs[:,1] = dts.bboxes[:,1] - dts.bboxes[:,3]/2
            bbs[:,2] = dts.bboxes[:,0] + dts.bboxes[:,2]/2
            bbs[:,3] = dts.bboxes[:,1] + dts.bboxes[:,3]/2
        elif dts._bb_format == 'x1y1x2y2':
            bbs = dts.bboxes.clone()
        else:
            raise NotImplementedError()

        scores, categories = dts.scores, dts.cats
        out_bbs, out_scores, out_cats = [], [], []
        unique_labels = categories.unique()
        for cat_idx in unique_labels:
            cls_mask = (categories==cat_idx)
            keep_indices = torchvision.ops.nms(bbs[cls_mask,:], scores[cls_mask], 
                                               nms_thres)
            out_bbs.append(dts.bboxes[cls_mask,:][keep_indices,:])
            out_scores.append(scores[cls_mask][keep_indices])
            out_cats.append(categories[cls_mask][keep_indices])
        
        out_bbs = torch.cat(out_bbs, dim=0)
        out_scores = torch.cat(out_scores, dim=0)
        out_cats = torch.cat(out_cats, dim=0)
        return ImageObjects(out_bbs, out_cats, out_scores, dts._bb_format)
    
    def draw_on_np(self, im, class_map='COCO', **kwargs):
        assert self.bboxes.dim() == 2
        if self.scores is not None:
            c = self.cats.float().unsqueeze(1)
            s = self.scores.unsqueeze(1)
            tmpbb = torch.cat([c, self.bboxes, s], dim=1)
            if class_map == 'COCO':
                draw_cocobb_on_np(im, tmpbb, bb_type='pbb', **kwargs)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def to_coco_json(self, img_id):
        list_json = []
        for bb, c, s in zip(self.bboxes, self.cats, self.scores):
            if self._bb_format == 'cxcywh':
                cx,cy,w,h = [float(t) for t in bb]
                bbox = [cx-w/2, cy-h/2, w, h]
            else:
                raise NotImplementedError()
            cat_id = COCO_CATEGORY_LIST[int(c)]['id']
            dt_dict = {'image_id': img_id, 'category_id': cat_id,
                       'bbox': bbox, 'score': float(s)}
            list_json.append(dt_dict)
        return list_json
