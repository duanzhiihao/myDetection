import torch
import torch.nn as nn
import torch.nn.functional as tnf

import models.losses as lossLib
from utils.bbox_ops import bboxes_iou
from utils.structures import ImageObjects


class DetectLayer(nn.Module):
    '''
    calculate the output boxes and losses
    '''
    def __init__(self, level_i: int, cfg: dict):
        super().__init__()
        anchors_all = torch.Tensor(cfg['model.detect.anchors'])
        indices     = cfg['model.detect.anchor_indices'][level_i]
        indices     = torch.Tensor(indices).long()

        self.indices = indices
        # anchors: tensor, e.g. shape(3,2), [[116, 90], [156, 198], [373, 326]]
        self.anchors = anchors_all[indices, :]
        # all anchors, rows of (0, 0, w, h), used for calculating IoU
        self.anch_00wh_all = torch.zeros(len(anchors_all), 4)
        self.anch_00wh_all[:,2:4] = anchors_all # unnormalized
        self.grid = torch.zeros(1) # dummy

        self.num_anchors      = len(indices)
        self.stride           = cfg['model.fpn.out_strides'][level_i]
        self.strides_all      = cfg['model.fpn.out_strides']
        self.n_cls            = cfg['general.num_class']
        self.sample_selection = cfg['model.detect.sample_selection']
        self.conf_target      = cfg['model.detect.confidence_target']
        self.negative_thres   = cfg.get('model.detect.negative_threshold', 0.7)
        self.loss_bbox        = cfg['model.detect.loss_bbox']
        self.bbox_format      = cfg['general.pred_bbox_format']

    def _make_grid(self, ny, nx, device):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        self.grid = torch.stack([xv, yv], dim=2).view(1, 1, ny, nx, 2).float()

    def forward(self, raw: dict, img_size, labels=None):
        assert isinstance(raw, dict)
        t_bbox = raw['bbox']
        device = t_bbox.device
        nB = t_bbox.shape[0] # batch size
        nA = self.num_anchors # number of anchors
        nH, nW = t_bbox.shape[2:4] # prediction grid size
        assert t_bbox.shape[1] == nA and t_bbox.shape[-1] == 4
        conf_logits = raw['conf']
        cls_logits = raw['class']

        if self.bbox_format == 'cxcywh':
            assert t_bbox.shape[-1] == 4
        elif self.bbox_format == 'cxcywhd':
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        # ----------------------- logits to prediction -----------------------
        p_bbox = t_bbox.detach().clone().contiguous()
        # bounding box
        if self.grid.shape[2:4] != (nH, nW):
            self._make_grid(nH, nW, device)
        self.grid = self.grid.to(device=device)
        p_bbox = torch.sigmoid(p_bbox)
        # x, y
        p_bbox[..., 0:2] = (p_bbox[..., 0:2] * 2 - 0.5 + self.grid) * self.stride
        # w, h
        anch_wh = self.anchors.view(1, nA, 1, 1, 2).to(device=device)
        p_bbox[..., 2:4] = (p_bbox[..., 2:4] * 2)**2 * anch_wh
        # angle
        if self.bbox_format == 'cxcywhd':
            raise NotImplementedError()
        bb_param = p_bbox.shape[-1]
        p_bbox = p_bbox.view(nB, nA*nH*nW, bb_param).cpu()

        # Logistic activation for confidence score
        p_conf = torch.sigmoid(conf_logits.detach())
        # Logistic activation for categories
        if self.n_cls > 0:
            p_cls = torch.sigmoid(cls_logits.detach())
        cls_score, cls_idx = torch.max(p_cls, dim=-1, keepdim=True)
        confs = p_conf * cls_score
        preds = {
            'bbox': p_bbox,
            'class_idx': cls_idx.view(nB, nA*nH*nW).cpu(),
            'score': confs.view(nB, nA*nH*nW).cpu(),
        }
        if labels is None:
            return preds, None

        if self.sample_selection == 'ATSS':
            raise NotImplementedError()
            # Build x,y meshgrid for all levels
            # Calculating this at each level is not very efficient
            # Ideally this should be done only once
            # But in order to achieve that, code structure must be changed.
            img_h, img_w = img_size
            all_anchor_bbs = []
            for li, s in enumerate(self.strides_all):
                assert img_w % s == 0 and img_h % s == 0
                _sdH, _sdW = img_h // s, img_w // s
                _x = torch.linspace(0, img_w, steps=_sdW+1)[:-1] + 0.5 * s
                _y = torch.linspace(0, img_h, steps=_sdH+1)[:-1] + 0.5 * s
                _gy, _gx = torch.meshgrid(_y, _x)
                # if s == stride:
                #     assert (_gy == gy).all() and (_gx == gx).all()
                assert _gy.shape == _gx.shape == (_sdH, _sdW)
                anch_wh = torch.ones(_sdH*_sdW, 2) * self.anchors_all[li]
                anch_bbs = torch.cat(
                    [_gx.reshape(-1,1), _gy.reshape(-1,1), anch_wh], dim=1)
                all_anchor_bbs.append(anch_bbs)

        assert isinstance(labels, list)
        valid_gt_num = 0
        TargetConf = torch.zeros(nB, nA, nH, nW, 1)
        if self.conf_target == 'zero-one':
            IgnoredMask = torch.zeros(nB, nA, nH, nW, dtype=torch.bool)
        loss_xy  = 0
        loss_wh  = 0
        loss_cls = 0
        bce_logits = tnf.binary_cross_entropy_with_logits
        # traverse all images in a batch
        for b in range(nB):
            im_labels = labels[b]
            im_labels: ImageObjects
            im_labels.sanity_check()
            num_gt = len(im_labels)
            if num_gt == 0:
                # no ground truth
                continue
            gt_bboxes = im_labels.bboxes
            gt_cls_idxs = im_labels.cats
            assert gt_bboxes.shape[1] == 4 # TODO:

            for gi, (gt_bb, gt_cidx) in enumerate(zip(gt_bboxes, gt_cls_idxs)):
                # --------------- find positive samples
                if self.sample_selection == 'best':
                    _gt_00wh = gt_bb.clone()
                    _gt_00wh[0:2] = 0
                    anchor_ious = bboxes_iou(_gt_00wh, self.anch_00wh_all, xyxy=False)
                    anch_idx_all = torch.argmax(anchor_ious, dim=1).squeeze().item()
                    if not (self.indices == anch_idx_all).any():
                        # this layer is not responsible for this GT
                        continue
                    ta = anch_idx_all % nA
                    ti = (gt_bb[0] / self.stride).long() # horizontal
                    tj = (gt_bb[1] / self.stride).long() # vertical
                    valid_gt_num += 1
                    # positive sample is (ta, tj, ti)
                elif self.sample_selection == 'ATSS':
                    raise NotImplementedError()
                else:
                    raise NotImplementedError()

                # loss for bounding box
                if self.loss_bbox == 'smooth_L1':
                    _t_bb = t_bbox[b, ta, tj, ti]
                    assert _t_bb.dim() == 1
                    _tgtxy  = ((gt_bb[:2] / self.stride) % 1 + 0.5) / 2
                    _tgtxy  = _tgtxy.to(device=device)
                    loss_xy = loss_xy + bce_logits(_t_bb[:2], _tgtxy, reduction='sum')
                    _tgtwh  = torch.sqrt(gt_bb[2:4] / self.anchors[ta,:]) / 2
                    _tgtwh  = _tgtwh.to(device=device)
                    loss_wh = loss_wh + bce_logits(_t_bb[2:4], _tgtwh, reduction='sum')
                    if _t_bb.shape[0] > 4:
                        raise NotImplementedError()
                elif self.loss_bbox == 'GIoU':
                    raise NotImplementedError()

                # loss for categories
                if self.n_cls > 0:
                    _t_cls = cls_logits[b, ta, tj, ti]
                    assert _t_cls.shape[-1] == self.n_cls
                    _tgt_cls = torch.zeros_like(_t_cls)
                    _tgt_cls[..., gt_cidx] = 1
                    loss_cls = loss_cls + bce_logits(_t_cls, _tgt_cls)
                
                # regression target for confidence score
                if self.conf_target == 'zero-one':
                    TargetConf[b, ta, tj, ti] = 1

            # loss for confidence score
            if self.bbox_format == 'cxcywh':
                pred_ious = bboxes_iou(p_bbox[b], gt_bboxes, xyxy=False)
            elif self.bbox_format == 'cxcywhd':
                raise NotImplementedError()
            iou_with_gt, _ = pred_ious.max(dim=1)
            if self.conf_target == 'IoU':
                TargetConf[b] = iou_with_gt.view(nA, nH, nW, 1)
            elif self.conf_target == 'zero-one':
                if self.bbox_format == 'cxcywh':
                    IgnoredMask[b] = (iou_with_gt > self.negative_thres).view(nA,nH,nW)
                elif self.bbox_format == 'cxcywhd':
                    raise NotImplementedError()

        # move the tagerts to GPU
        TargetConf = TargetConf.to(device=device)
        ignored_num = 0
        if self.conf_target == 'IoU':
            loss_conf = bce_logits(conf_logits, TargetConf, reduction='sum')
        elif self.conf_target == 'zero-one':
            IgnoredMask = IgnoredMask.to(device=device)
            _pos_mask = TargetConf.squeeze(-1).bool()
            _penalty = _pos_mask | (~IgnoredMask)
            loss_conf = bce_logits(conf_logits[_penalty], TargetConf[_penalty],
                                   reduction='sum')
            ignored_num = (IgnoredMask & (~_pos_mask)).sum().cpu().item()
        else:
            raise NotImplementedError()

        loss = loss_xy + loss_wh + loss_conf + loss_cls
        loss = loss / nB

        # logging
        ngt = valid_gt_num + 1e-16
        self.loss_str = f'yolo_{nH}x{nW} pos/ignore: {int(ngt)}/{ignored_num}: ' \
                        f'xy/gt {loss_xy/ngt:.3f}, wh/gt {loss_wh/ngt:.3f}, ' \
                        f'conf {loss_conf:.3f}, class {loss_cls:.3f}'
        self._assigned_num = valid_gt_num
        return preds, loss
