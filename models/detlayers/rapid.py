import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tnf

import models.losses as lossLib
from utils.bbox_ops import iou_rle
from utils.structures import ImageObjects


class RAPiDLayer(nn.Module):
    '''
    calculate the output boxes and losses
    '''
    def __init__(self, level_i: int, cfg: dict):
        super().__init__()
        anchors_all = torch.Tensor(cfg['model.rapid.anchors'])
        indices = cfg['model.rapid.anchor_indices'][level_i]
        indices = torch.Tensor(indices).long()
        self.anchor_indices = indices
        # anchors: tensor, e.g. shape(3,2), [[116, 90], [156, 198], [373, 326]]
        self.anchors = anchors_all[indices, :]
        # all anchors, rows of (0, 0, w, h, a). Used for calculating IoU
        self.anch_00wha_all = torch.zeros(len(anchors_all), 5)
        self.anch_00wha_all[:,2:4] = anchors_all # unnormalized
        self.num_anchors = len(indices)
        self.stride = cfg['model.fpn.out_strides'][level_i]
        self.n_cls = cfg['general.num_class']

        self.ignore_thre = 0.6
        assert cfg.get('model.angle.pred_range', 360) == 360
        # loss function setting
        self.wh_sl1_beta = cfg.get('model.rapid.wh_smooth_l1_beta')
        self.loss4angle = lossLib.get_angle_loss(cfg['model.angle.loss_angle'], 'sum')
        self.loss4angle = nn.MSELoss(reduction='sum')

    def forward(self, raw: dict, img_size, labels=None):
        assert isinstance(raw, dict)
        t_xywha = raw['bbox']
        device = t_xywha.device
        nB = t_xywha.shape[0] # batch size
        nA = self.num_anchors # number of anchors
        nH, nW = t_xywha.shape[2:4] # prediction grid size
        assert t_xywha.shape[1] == nA and t_xywha.shape[-1] == 5
        conf_logits = raw['conf']
        cls_logits = raw['class']
        
        # Convert the predicted angle to radian for later use.
        p_radian = torch.sigmoid(t_xywha[..., 4]) * 2 * np.pi - np.pi

        # ----------------------- logits to prediction -----------------------
        p_xywha = t_xywha.detach().clone()
        # sigmoid activation for xy, angle, obj_conf
        y_ = torch.arange(nH, dtype=torch.float, device=device)
        x_ = torch.arange(nW, dtype=torch.float, device=device)
        mesh_y, mesh_x = torch.meshgrid(y_, x_)
        p_xywha[..., 0] = (torch.sigmoid(p_xywha[..., 0]) + mesh_x) * self.stride
        p_xywha[..., 1] = (torch.sigmoid(p_xywha[..., 1]) + mesh_y) * self.stride
        # w, h
        anch_wh = self.anchors.view(1,nA,1,1,2).to(device=device)
        p_xywha[..., 2:4] = torch.exp(p_xywha[..., 2:4]) * anch_wh
        p_xywha[..., 4] = p_radian / np.pi * 180
        p_xywha = p_xywha.cpu()
        
        if labels is None:
            # Logistic activation for confidence score
            p_conf = torch.sigmoid(conf_logits)
            # Logistic activation for categories
            if self.n_cls > 0:
                p_cls = torch.sigmoid(cls_logits)
                cls_score, cls_idx = torch.max(p_cls, dim=-1, keepdim=True)
                confs = p_conf * cls_score
                preds = {
                    'bbox': p_xywha.view(nB, nA*nH*nW, 5),
                    'class_idx': cls_idx.view(nB, nA*nH*nW).cpu(),
                    'score': confs.view(nB, nA*nH*nW).cpu(),
                }
            else:
                preds = {
                    'bbox': p_xywha.view(nB, nA*nH*nW, 5),
                    'class_idx': torch.zeros(nB, nA*nH*nW, dtype=torch.int64),
                    'score': p_conf.view(nB, nA*nH*nW).cpu(),
                }
            return preds, None

        assert isinstance(labels, list)
        self.valid_gts = []
        # Initialize prediction targets
        PositiveMask = torch.zeros(nB, nA, nH, nW, dtype=torch.bool)
        IgnoredMask = torch.zeros(nB, nA, nH, nW, dtype=torch.bool)
        weighted = torch.zeros(nB, nA, nH, nW)
        TargetXYWH = torch.zeros(nB, nA, nH, nW, 4)
        TargetAngle = torch.zeros(nB, nA, nH, nW)
        TargetConf = torch.zeros(nB, nA, nH, nW, 1)
        if self.n_cls > 0:
            TargetCls = torch.zeros(nB, nA, nH, nW, self.n_cls)
        
        # traverse all images in a batch
        for b in range(nB):
            im_labels = labels[b]
            assert isinstance(im_labels, ImageObjects)
            assert im_labels._bb_format == 'cxcywhd'
            assert im_labels.img_hw == img_size
            im_labels.sanity_check()
            nGT = len(im_labels)

            # if there is no ground truth, continue
            if nGT == 0:
                continue

            gt_bboxes = im_labels.bboxes
            gt_cls_idxs = im_labels.cats
            assert gt_bboxes.shape[1] == 5 # (cx, cy, w, h, degree)

            # IoU between all predicted bounding boxes and all GT.
            # rot bbox IoU calculation is expensive so only calculate IoU
            # using confident samples
            selected_idx = conf_logits[b] > - np.log(1/(0.005) - 1)
            selected_idx = selected_idx.squeeze(-1)
            p_selected = p_xywha[b][selected_idx]
            # if all predictions are lower than 0.005, penalize everything
            # if too many preictions are higher than 0.005, penalize everything
            if len(p_selected) < 1000 and len(p_selected) > 0:
                pred_ious = iou_rle(p_selected.view(-1,5), gt_bboxes,
                                    bb_format='cxcywhd', img_size=img_size)
                iou_with_gt, _ = pred_ious.max(dim=1)
                # do not penalize the predicted bboxes who have a high overlap
                # with any GT
                IgnoredMask[b, selected_idx] = (iou_with_gt > self.ignore_thre)
                # conf_loss_mask == 1 means give penalty

            for gi, (gt_bb, gt_cidx) in enumerate(zip(gt_bboxes, gt_cls_idxs)):
                assert gt_bb.dim() == 1 and len(gt_bb) == 5
                _gt_00wh0 = gt_bb.clone()
                _gt_00wh0[0:2] = 0
                _gt_00wh0[4] = 0
                _gt_00wh0 = _gt_00wh0.unsqueeze(0)
                anchor_ious = iou_rle(_gt_00wh0, self.anch_00wha_all,
                                    bb_format='cxcywhd', img_size=img_size)
                anch_idx_all = torch.argmax(anchor_ious, dim=1).squeeze().item()
                if not (self.anchor_indices == anch_idx_all).any():
                    # this layer is not responsible for this GT
                    continue

                self.valid_gts.append(gt_bb.clone())
                ta = anch_idx_all % nA # target anchor index
                ti = (gt_bb[0] / self.stride).long() # horizontal
                tj = (gt_bb[1] / self.stride).long() # vertical

                # positive sample
                PositiveMask[b,ta,tj,ti] = True
                # target x, y
                TargetXYWH[b,ta,tj,ti,0] = (gt_bb[0] / self.stride) % 1
                TargetXYWH[b,ta,tj,ti,1] = (gt_bb[1] / self.stride) % 1
                TargetXYWH[b,ta,tj,ti,2] = torch.log(gt_bb[2]/self.anchors[ta,0] + 1e-8)
                TargetXYWH[b,ta,tj,ti,3] = torch.log(gt_bb[3]/self.anchors[ta,1] + 1e-8)
                # use radian when calculating angle loss
                TargetAngle[b,ta,tj,ti] = gt_bb[4] / 180 * np.pi
                # target confidence score
                TargetConf[b,ta,tj,ti] = 1
                # target category
                if self.n_cls > 0:
                    TargetCls[b, ta, tj, ti, gt_cidx] = 1
                # smaller objects have higher losses
                img_area = img_size[0] * img_size[1]
                weighted[b,ta,tj,ti] = 2 - gt_bb[2] * gt_bb[3] / img_area

        # move the tagerts to GPU
        PositiveMask = PositiveMask.to(device=device)
        IgnoredMask = IgnoredMask.to(device=device)
        TargetXYWH = TargetXYWH.to(device=device)
        TargetAngle = TargetAngle.to(device=device)
        TargetConf = TargetConf.to(device=device)
        if self.n_cls > 0:
            TargetCls = TargetCls.to(device=device)
        weighted = weighted.unsqueeze(-1).to(device=device)[PositiveMask]
        
        bce_logits = tnf.binary_cross_entropy_with_logits
        # weighted *BCE* loss for xy
        _pxy = t_xywha[...,0:2][PositiveMask]
        _tgtxy = TargetXYWH[...,0:2][PositiveMask]
        loss_xy = bce_logits(_pxy, _tgtxy, weight=weighted, reduction='sum')
        # weighted loss for w,h
        _pwh = t_xywha[...,2:4][PositiveMask]
        _tgtwh = TargetXYWH[...,2:4][PositiveMask]
        # loss_wh = 0.5 * (weighted * (_pwh - _tgtwh).pow(2)).sum()
        weighted = torch.cat([weighted,weighted], dim=1)
        loss_wh = lossLib.smooth_L1_loss(_pwh, _tgtwh, beta=self.wh_sl1_beta,
                                         weight=weighted, reduction='sum')
        # loss for angle
        loss_angle = self.loss4angle(p_radian[PositiveMask], TargetAngle[PositiveMask])
        # confidence score
        _penalty = PositiveMask | (~IgnoredMask)
        loss_conf = bce_logits(conf_logits[_penalty], TargetConf[_penalty],
                              reduction='sum')
        if self.n_cls > 0:
            loss_cls = bce_logits(cls_logits[PositiveMask], TargetCls[PositiveMask], 
                                  reduction='sum')
        else:
            loss_cls = 0
        loss = loss_xy + loss_wh + loss_angle + loss_conf + loss_cls
        loss = loss / nB

        # logging
        pos_num = PositiveMask.sum().cpu().item()
        ngt = pos_num + 1e-16
        ignored_num = (IgnoredMask & (~PositiveMask)).sum().cpu().item()
        self.loss_str = f'level_{nH}x{nW} pos/ignore: {int(ngt)}/{ignored_num}, loss: ' \
                        f'xy/gt {loss_xy/ngt:.3f}, wh/gt {loss_wh/ngt:.3f}, ' \
                        f'angle/gt {loss_angle/ngt:.3f}, conf {loss_conf:.3f}, ' \
                        f'class {loss_cls:.3f}'
        self._assigned_num = pos_num
        return None, loss
