import numpy as np
import torch
import torchvision

from .bbox_ops import nms_rotbb
from . import tracking as trackUtils
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
        img_hw: int or (height, width)
    '''
    def __init__(self, bboxes, cats, scores=None,
                       bb_format='cxcywh', img_hw=None):
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
        self.img_hw = img_hw
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            b = self.bboxes[idx:idx+1,:]
            c = self.cats[idx:idx+1]
            s = self.scores[idx:idx+1] if self.scores is not None else None
            return ImageObjects(b, c, s, self._bb_format, self.img_hw)
        b = self.bboxes[idx,:]
        c = self.cats[idx]
        s = self.scores[idx] if self.scores is not None else None
        return ImageObjects(b, c, s, self._bb_format, self.img_hw)
    
    def __len__(self):
        obj_num = self.bboxes.shape[0]
        return obj_num
    
    def cpu_(self):
        self.bboxes = self.bboxes.cpu()
        self.cats = self.cats.cpu()
        self.scores = self.scores.cpu() if self.scores is not None else None

    def sort_by_score_(self, descending=True):
        '''
        Sort the bounding boxes by scores in-place
        '''
        assert self.scores is not None
        idxs = torch.argsort(self.scores, descending=descending)
        self.bboxes = self.bboxes[idxs, :]
        self.cats = self.cats[idxs]
        self.scores = self.scores[idxs]

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
            _bbs = dts.bboxes.clone()
            _bbs[:,0] = dts.bboxes[:,0] - dts.bboxes[:,2]/2
            _bbs[:,1] = dts.bboxes[:,1] - dts.bboxes[:,3]/2
            _bbs[:,2] = dts.bboxes[:,0] + dts.bboxes[:,2]/2
            _bbs[:,3] = dts.bboxes[:,1] + dts.bboxes[:,3]/2
            single_cls_nms_func = torchvision.ops.nms
        elif dts._bb_format == 'x1y1x2y2':
            _bbs = dts.bboxes.clone()
            single_cls_nms_func = torchvision.ops.nms
        elif dts._bb_format == 'cxcywhd':
            _bbs = dts.bboxes.clone()
            def single_cls_nms_func(boxes, scores, nms_thres):
                img_hw = dts.img_hw or 1024
                return nms_rotbb(boxes, scores, nms_thres, bb_format=dts._bb_format,
                                 img_size=img_hw, majority=None)
        else:
            raise NotImplementedError()

        scores, categories = dts.scores, dts.cats
        # keep_idxs = single_cls_nms_func(_bbs, scores, nms_thres)
        # out_bbs = dts.bboxes[keep_idxs, :]
        # out_scores = scores[keep_idxs]
        # out_cats = categories[keep_idxs]
        out_bbs, out_scores, out_cats = [], [], []
        unique_labels = categories.unique()
        for cat_idx in unique_labels:
            cls_mask = (categories==cat_idx)

            keep_indices = single_cls_nms_func(_bbs[cls_mask,:], scores[cls_mask],
                                               nms_thres)

            out_bbs.append(dts.bboxes[cls_mask,:][keep_indices,:])
            out_scores.append(scores[cls_mask][keep_indices])
            out_cats.append(categories[cls_mask][keep_indices])
        
        out_bbs = torch.cat(out_bbs, dim=0)
        out_scores = torch.cat(out_scores, dim=0)
        out_cats = torch.cat(out_cats, dim=0)
        return ImageObjects(out_bbs, out_cats, out_scores, dts._bb_format,
                            img_hw=dts.img_hw)
    
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
        self.img_hw = (ori_h, ori_w)
    
    def sanity_check(self):
        # Check dimension and length
        assert self.bboxes.dim() == 2 and self.cats.dim() == 1
        assert self.cats.shape[0] == self.bboxes.shape[0]
        if self.scores is not None:
            assert self.scores.shape[0] == self.bboxes.shape[0]
        # Check bbox format
        if self._bb_format == 'cxcywh':
            assert self.bboxes.shape[1] == 4
        elif self._bb_format == 'cxcywhd':
            assert self.bboxes.shape[1] == 5
        else:
            raise NotImplementedError()
        # Check dtype
        assert self.cats.dtype == torch.int64, 'Incorrect data type of categories'
        # Check image size
        assert self.img_hw is None or isinstance(self.img_hw, (int,list,tuple,torch.Size))
    
    def draw_on_np(self, im, class_map='COCO', **kwargs):
        assert self.bboxes.dim() == 2
        draw_bboxes_on_np(im, self, class_map=class_map, **kwargs)

    def to_json(self, img_id):
        assert self.bboxes.dim() == 2
        assert self.bboxes.shape[0] == self.cats.shape[0] == self.scores.shape[0]
        list_json = []
        for bb, c, s in zip(self.bboxes, self.cats, self.scores):
            if self._bb_format == 'cxcywh':
                cx,cy,w,h = [float(t) for t in bb]
                bbox = [cx-w/2, cy-h/2, w, h]
            elif self._bb_format == 'cxcywhd':
                bbox = [float(t) for t in bb]
            else:
                raise NotImplementedError()
            cat_id = COCO_CATEGORY_LIST[int(c)]['id']
            dt_dict = {'image_id': img_id, 'category_id': cat_id,
                       'bbox': bbox, 'score': float(s)}
            list_json.append(dt_dict)
        return list_json


class OnlineTracklet():
    '''
    Args:
        prev_len: int, previous length that will be filled with None
        cur_obj: ImageObjects, should have a length of 1,
                 representing one object in the current image
        buf_len: int, buffer length. Each tracklet will reserve some buffer \
                 in order not to frequently call the torch.cat method.
    '''
    def __init__(self, prev_len: int, cur_obj: ImageObjects,
                 buf_len: int=100, color=None, obj_id=None):
        assert isinstance(cur_obj, ImageObjects) and len(cur_obj) == 1

        # list of past and current bounding boxes
        self._bboxes = torch.zeros(prev_len+buf_len, cur_obj.bboxes.shape[1])
        self._bboxes[prev_len, :] = cur_obj.bboxes.squeeze(0)
        # list of past and currrent scores (if any)
        if cur_obj.scores is None:
            self._scores = None
        else:
            self._scores = torch.zeros(prev_len+buf_len)
            self._scores[prev_len] = cur_obj.scores
        # self.length indicates the current valid length of the tracklet
        self.length = prev_len + 1
        # attributes that shoud apply to all the frames
        self.category = cur_obj.cats[0] # torch.LongTensor
        self._bb_format = cur_obj._bb_format
        self.img_hw = cur_obj.img_hw

        self.buf_len = buf_len
        if color is not None:
            self.color = color
        else:
            self.color = tuple([np.random.randint(256) for _ in range(3)])
        # self.sanity_check()
        self._pred_count = 0
        assert obj_id is not None
        self.obj_id = obj_id
    
    def get_bboxes(self):
        return self._bboxes[:self.length, :]
    
    def get_scores(self):
        return self._scores[:self.length, :]

    def __len__(self):
        return self.length
    
    def append(self, cur_obj: ImageObjects):
        assert isinstance(cur_obj, ImageObjects) and len(cur_obj) == 1
        assert cur_obj._bb_format == self._bb_format
        assert self._pred_count > 0
        if cur_obj.img_hw is not None:
            if self.img_hw is not None:
                assert cur_obj.img_hw == self.img_hw
            else:
                self.img_hw = cur_obj.img_hw
        assert cur_obj.cats == self.category

        # check if the buffer overflows
        if self.length >= self._bboxes.shape[0]:
            self._bboxes = torch.cat([
                self._bboxes, torch.zeros(self.buf_len, self._bboxes.shape[1])
            ], dim=0)
            if self._scores is not None:
                self._scores = torch.cat([self._scores, torch.zeros(self.buf_len)])

        self._bboxes[self.length-1, :] = cur_obj.bboxes.squeeze(0)
        if cur_obj.scores is None:
            assert self._scores is None
        else:
            self._scores[self.length-1] = cur_obj.scores

        # update self.length
        # self.length += 1
        self._pred_count = 0
    
    def predict(self, xy_mode=None, wh_mode=None, angle_mode=None):
        assert self._bb_format in {'cxcywh', 'cxcywhd'}
        _bbs = self.get_bboxes()
        last_appear_idx = (_bbs.prod(dim=1) != 0).nonzero()[0]
        _bbs = _bbs[last_appear_idx:, :]
        if xy_mode == wh_mode == angle_mode == 'linear':
            pred_bb = trackUtils.linear_predict(_bbs)
        else:
            raise NotImplementedError()
        if self._bb_format == 'cxcywhd':
            pred_bb[4] = pred_bb[4] % 180
        self._pred_count += 1

        # check if the buffer overflows
        if self.length >= self._bboxes.shape[0]:
            self._bboxes = torch.cat([
                self._bboxes, torch.zeros(self.buf_len, self._bboxes.shape[1])
            ], dim=0)
            if self._scores is not None:
                self._scores = torch.cat([self._scores, torch.zeros(self.buf_len)])
        
        self._bboxes[self.length, :] = pred_bb
        self.length += 1
        return pred_bb

    def is_feasible(self):
        imh, imw = self.img_hw
        bbox = self._bboxes[self.length-1, :]
        if (bbox[:4] < 0).any():
            return False
        if (bbox[0] > imw) or (bbox[1] > imh) or (bbox[2] > imw) or (bbox[3] > imh):
            return False
        return True

    def sanity_check(self, history_len: int=None):
        raise NotImplementedError()

        # if history_len is None:
        #     # check all data
        #     _bb2check = self.bboxes
        #     _score2check = self.scores
        # else:
        #     # check recent data only
        #     assert isinstance(history_len, int)
        #     history_len = int(history_len) # just to pass the pylint check
        #     _bb2check = self.bboxes[-history_len:]
        #     _score2check = self.scores[-history_len:]
        # assert len(_bb2check) == len(_score2check)

        # # check bounding box format
        # if self._bb_format == 'cxcywh':
        #     bb_param = 4
        # elif self._bb_format == 'cxcywhd':
        #     bb_param = 5
        # else: raise NotImplementedError()
        # assert all(filter(lambda b: b is None or len(b)==bb_param, _bb2check))

        # if there are scores, num of score must equal to num of bounding box
        # if any(filter(lambda x: x is not None, _score2check)):
        #     assert all(map(lambda bs: (bs[0] is None)==(bs[1] is None), 
        #                    zip(_bb2check,_score2check)))

        # Check image size
        # assert self.img_hw is None or isinstance(self.img_hw, (int,list,tuple,torch.Size))


# Q_cov=[0.05, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01],
# R_cov=[0.01, 0.01, 0.05, 0.05]

# Edge_cases:
# Q = [0.04929452, 0.03181672, 0.05152841, 0.09702075, 13.61979758]
# R = [0.07263693, 0.06385201, 0.12444371, 0.16264272, 24.38639988]
from .kalman_filter import RotBBoxKalmanFilter
class KFTracklet():
    '''
    Tracklet with Kalman Filter (KF).

    The notation of the KF parameters are consistent with the wikipedia page: \
    https://en.wikipedia.org/wiki/Kalman_filter

    Args:
        bbox: initial bounding box
    '''
    def __init__(self, bbox, score, object_id, global_step=0, img_hw=None):
        assert isinstance(bbox, np.ndarray) and bbox.shape[0] == 5
        self.kf = RotBBoxKalmanFilter(
            initial_P_cov=[0.1, 0.1, 0.1, 0.1, 10, 0.1, 0.1, 0.1, 0.1, 10],
            Q_cov=[0.049, 0.032, 0.052, 0.097, 13.62, 0.01, 0.01, 0.01, 0.01, 1],
            R_cov=[0.073, 0.064, 0.124, 0.163, 24.39])
        # self.kf.initiate(bbox[:4])
        bbox[4] = bbox[4] % 180
        self.kf.initiate(bbox)
        self.bbox = bbox
        self.score = score

        self.object_id = object_id
        self.step = global_step
        self.img_hw = img_hw
        self.momentum = 0.8
        self._pred_count = 0

    def predict(self):
        # cxcywh = self.kf.predict()
        # bbox = np.r_[cxcywh, self.predict_angle()]
        bbox = self.kf.predict()
        self.kf.x[4] = self.kf.x[4] % 180
        self.bbox = bbox

        self.step += 1
        if self._pred_count >= 1:
            self.score = self.momentum*self.score # + (1-self.momentum) * 0
        self._pred_count += 1
        return bbox.copy()
    
    def update(self, bbox, score) -> np.ndarray:
        assert isinstance(bbox, np.ndarray) and bbox.shape[0] == 5
        assert self._pred_count > 0, 'Please call predict() before update()'
        # cxcywh = self.kf.update(bbox[:4])
        # bbox = np.r_[cxcywh, bbox[4]]
        bbox = bbox.copy()
        z = bbox[4] % 180 # measurement
        angle_state = self.kf.x[4]
        assert 0 <= angle_state < 180
        bbox[4] = min(z, z-180, z+180, key=lambda x:abs(x-angle_state))
        bbox = self.kf.update(bbox)
        self.kf.x[4] = self.kf.x[4] % 180
        self.bbox = bbox
        self.score = self.momentum*self.score + (1-self.momentum)*score

        self._pred_count = 0
        return bbox.copy()

    def is_feasible(self):
        imh, imw = self.img_hw
        bbox = self.bbox
        if self.score < 0.1:
            return False
        if (bbox[:4] < 0).any():
            return False
        if (bbox[0] > imw) or (bbox[1] > imh) or (bbox[2] > imw) or (bbox[3] > imh):
            return False
        return True
    
    # def predict_angle(self):
    #     if len(self._angles) == 1:
    #         return self._angles[-1]
    #     return 2*self._angles[-1] - self._angles[-2]

    def likelihood(self, xywha: np.ndarray):
        assert xywha.ndim == 2 and xywha.shape[1] == 5
        mean = self.kf.x[:5].reshape(1,5)
        cov = self.kf.P[:5,:5]

        numerator = (xywha-mean) @ np.linalg.inv(cov)
        numerator = (numerator * (xywha-mean)).sum(axis=1)
        denom = np.sqrt((2*np.pi)**5 * np.linalg.det(cov))
        p = np.exp(-0.5 *  numerator) / denom
        return p
