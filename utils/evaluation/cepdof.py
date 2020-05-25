import json
import numpy as np
from collections import defaultdict

import cv2
from pycocotools import cocoeval
from pycocotools import mask as maskUtils


def evaluate_json(dts_json, gt_json_path):
    print('Initialing validation set...')
    cocoEval = CEPDOFeval(gt_json_path, dts_json, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    ap, ap50, ap75 = cocoEval.stats[0], cocoEval.stats[1], cocoEval.stats[2]
    # TODO: return a dict
    return cocoEval.summary, ap, ap50, ap75


class CEPDOFeval(cocoeval.COCOeval):
    def __init__(self, gt_json, dt_json, iouType='bbox'):
        assert iouType == 'bbox', 'Only support (rotated) bbox iou type'
        self.gt_json = json.load(open(gt_json, 'r')) if isinstance(gt_json, str) \
                       else gt_json
        self.dt_json = json.load(open(dt_json, 'r')) if isinstance(dt_json, str) \
                       else dt_json
        self._preprocess_dt_gt()
        self.params = cocoeval.Params(iouType=iouType)
        self.params.imgIds = sorted([img['id'] for img in self.gt_json['images']])
        self.params.catIds = sorted([cat['id'] for cat in self.gt_json['categories']])
        # Initialize some variables which will be modified later
        self.evalImgs = defaultdict(list)   # per-image per-category eval results
        self.eval     = {}                  # accumulated evaluation results
    
    def _preprocess_dt_gt(self):
        # We are not using 'id' in ground truth annotations because it's useless.
        # However, COCOeval API requires 'id' in both detections and ground truth.
        # So, add id to each dt and gt in the dt_json and gt_json.
        for i, gt in enumerate(self.gt_json['annotations']):
            gt['id'] = gt.get('id', i+1)
            gt['area'] = gt.get('area', gt['bbox'][2]*gt['bbox'][3])
        for i, dt in enumerate(self.dt_json):
            dt['id'] = dt.get('id', i+1)
            # Calculate the areas of detections if there is not. category_id
            dt['area'] = dt.get('area', dt['bbox'][2]*dt['bbox'][3])
            dt['category_id'] = dt.get('category_id', 1)
        # A dictionary mapping from image id to image information
        self.imgId_to_info = {img['id']:img for img in self.gt_json['images']}

    def _prepare(self):
        p = self.params
        gts = [ann for ann in self.gt_json['annotations']]
        dts = self.dt_json

        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt.get('ignore', False) or gt.get('iscrowd', False)
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            raise NotImplementedError()
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
            # get image width and height
            img = self.imgId_to_info[imgId]
            img_size = (img['height'], img['width'])
            iou_func = lambda x,y: iou_rle(x, y, img_size=img_size)
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        # iscrowd = [int(o['iscrowd']) for o in gt]
        ious = iou_func(d, g)
        return ious

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        self.summary = ''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            summ_str = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
            print(summ_str)
            self.summary += summ_str + '\n'
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()


def xywha2vertex(box, is_degree, stack=True):
    '''
    Args:
        box: tensor, shape(batch,5), 5=(cx, cy, w, h, degree)

    Return:
        tensor, shape(batch,4,2): topleft, topright, bottomright, bottomleft
    '''
    assert is_degree == False and box.ndim == 2 and box.shape[1] >= 5
    batch = box.shape[0]

    center = box[:,0:2]
    w = box[:,2]
    h = box[:,3]
    rad = box[:,4]
    # calculate vertical vector
    verti = np.empty((batch,2))
    verti[:,0] = (h/2) * np.sin(rad)
    verti[:,1] = - (h/2) * np.cos(rad)
    # calculate horizontal vector
    hori = np.empty((batch,2))
    hori[:,0] = (w/2) * np.cos(rad)
    hori[:,1] = (w/2) * np.sin(rad)
    # calculate four vertices
    tl = center + verti - hori
    tr = center + verti + hori
    br = center - verti + hori
    bl = center - verti - hori

    return np.concatenate([tl,tr,br,bl], axis=1)


def iou_rle(boxes1, boxes2, img_size=2048):
    '''
    Use mask and Run Length Encoding to calculate IOU between rotated bboxes.

    NOTE: rotated bounding boxes format is [cx, cy, w, h, degree].

    Args:
        boxes1: list[list[float]], shape[M,5], 5=(cx, cy, w, h, degree)
        boxes2: list[list[float]], shape[N,5], 5=(cx, cy, w, h, degree)
        img_size: int or list, (height, width)

    Return:
        ious: np.array[M,N], ious of all bounding box pairs
    '''
    assert isinstance(boxes1, list) and isinstance(boxes2, list)
    # convert bounding boxes to torch.tensor
    boxes1 = np.array(boxes1).reshape(-1, 5)
    boxes2 = np.array(boxes2).reshape(-1, 5)
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]))
    
    # Convert angle from degree to radian
    boxes1[:,4] = boxes1[:,4] * np.pi / 180
    boxes2[:,4] = boxes2[:,4] * np.pi / 180

    b1 = xywha2vertex(boxes1, is_degree=False).tolist()
    b2 = xywha2vertex(boxes2, is_degree=False).tolist()
    
    h, w = (img_size, img_size) if isinstance(img_size, int) else img_size
    b1 = maskUtils.frPyObjects(b1, h, w)
    b2 = maskUtils.frPyObjects(b2, h, w)
    ious = maskUtils.iou(b1, b2, [0 for _ in b2])

    return ious


def draw_cxcywhd(im, cx, cy, w, h, degree, color=(255,0,0), linewidth=5):
    '''
    Draw a rotated bounding box on an np-array image in-place.

    Args:
        im: image numpy array, shape(h,w,3)
        cx, cy, w, h: the center x, center y, width, and height of the rot bbox
        degree: the angle that the bbox is rotated clockwise
    '''
    c, s = np.cos(degree/180*np.pi), np.sin(degree/180*np.pi)
    R = np.asarray([[c, s], [-s, c]])
    pts = np.asarray([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    rot_pts = []
    for pt in pts:
        rot_pts.append(([cx, cy] + pt @ R).astype(int))
    contours = np.array([rot_pts[0], rot_pts[1], rot_pts[2], rot_pts[3]])
    cv2.polylines(im, [contours], isClosed=True, color=color,
                thickness=linewidth, lineType=cv2.LINE_AA)