import torch
import torch.nn as nn
import torch.nn.functional as tnf

import models.losses as lossLib
from utils.bbox_ops import bboxes_iou


class YOLOLayer(nn.Module):
    '''
    calculate the output boxes and losses
    '''
    def __init__(self, level_i: int, cfg: dict):
        super().__init__()
        anchors_all = torch.Tensor(cfg['model.yolo.anchors'])
        indices = cfg['model.yolo.anchor_indices'][level_i]
        indices = torch.Tensor(indices).long()
        self.indices = indices
        # anchors: tensor, e.g. shape(3,2), [[116, 90], [156, 198], [373, 326]]
        self.anchors = anchors_all[indices, :]
        # all anchors, rows of (0, 0, w, h), used for calculating IoU
        self.anch_00wh_all = torch.zeros(len(anchors_all), 4)
        self.anch_00wh_all[:,2:4] = anchors_all # unnormalized

        self.ignore_thre = cfg['model.yolo.anchor.negative_threshold']
        self.num_anchors = len(indices)
        self.stride = cfg['model.fpn.out_strides'][level_i]
        self.n_cls = cfg['general.num_class']

    def forward(self, raw: dict, img_size, labels=None):
        assert isinstance(raw, dict)
        t_xywh = raw['bbox']
        device = t_xywh.device
        nB = t_xywh.shape[0] # batch size
        nA = self.num_anchors # number of anchors
        nH, nW = t_xywh.shape[2:4] # prediction grid size
        assert t_xywh.shape[1] == nA and t_xywh.shape[-1] == 4
        conf_logits = raw['conf']
        cls_logits = raw['class']

        # ----------------------- logits to prediction -----------------------
        p_xywh = t_xywh.detach().clone().contiguous()
        # sigmoid activation for xy, obj_conf
        y_ = torch.arange(nH, dtype=torch.float, device=device).view(1,1,nH,1)
        x_ = torch.arange(nW, dtype=torch.float, device=device).view(1,1,1,nW)
        p_xywh[..., 0] = (torch.sigmoid(p_xywh[..., 0]) + x_) * self.stride
        p_xywh[..., 1] = (torch.sigmoid(p_xywh[..., 1]) + y_) * self.stride
        # w, h
        anch_wh = self.anchors.view(1,nA,1,1,2).to(device=device)
        p_xywh[...,2:4] = torch.exp(p_xywh[...,2:4]) * anch_wh
        p_xywh = p_xywh.view(nB, nA*nH*nW, 4).cpu()

        # Logistic activation for confidence score
        p_conf = torch.sigmoid(conf_logits.detach())
        # Logistic activation for categories
        if self.n_cls > 0:
            p_cls = torch.sigmoid(cls_logits.detach())
            cls_score, cls_idx = torch.max(p_cls, dim=-1, keepdim=True)
            confs = p_conf * cls_score
        else:
            cls_idx = torch.zeros(nB, nA, nH, nW, dtype=torch.int64)
            confs = p_conf
        preds = {
            'bbox': p_xywh,
            'class_idx': cls_idx.view(nB, nA*nH*nW).cpu(),
            'score': confs.view(nB, nA*nH*nW).cpu(),
        }
        if labels is None:
            return preds, None
            
        assert isinstance(labels, list)
        valid_gt_num = 0
        gt_mask = torch.zeros(nB, nA, nH, nW, dtype=torch.bool)
        conf_loss_mask = torch.ones(nB, nA, nH, nW, dtype=torch.bool)
        weighted = torch.zeros(nB, nA, nH, nW)
        tgt_xywh = torch.zeros(nB, nA, nH, nW, 4)
        tgt_conf = torch.zeros(nB, nA, nH, nW, 1)
        tgt_cls = torch.zeros(nB, nA, nH, nW, self.n_cls)
        # traverse all images in a batch
        for b in range(nB):
            im_labels = labels[b]
            im_labels.sanity_check()
            num_gt = len(im_labels)
            if num_gt == 0:
                # no ground truth
                continue
            gt_bboxes = im_labels.bboxes
            gt_cls_idx = im_labels.cats
            assert gt_bboxes.shape[1] == 4

            # calculate iou between truth and reference anchors
            gt_00wh = torch.zeros(num_gt, 4)
            gt_00wh[:, 2:4] = gt_bboxes[:, 2:4]
            anchor_ious = bboxes_iou(gt_00wh, self.anch_00wh_all, xyxy=False)
            best_n_all = torch.argmax(anchor_ious, dim=1)
            best_n = best_n_all % self.num_anchors
            
            valid_mask = torch.zeros(num_gt, dtype=torch.bool)
            for ind in self.indices:
                valid_mask = ( valid_mask | (best_n_all == ind) )
            if valid_mask.sum() == 0:
                # no anchor is responsible for any ground truth
                continue
            else:
                valid_gt_num += sum(valid_mask)

            pred_ious = bboxes_iou(p_xywh[b], gt_bboxes, xyxy=False)
            iou_with_gt, _ = pred_ious.max(dim=1)
            # ignore the conf of a pred BB if it matches a gt more than 0.7
            conf_loss_mask[b] = (iou_with_gt < self.ignore_thre).view(nA,nH,nW)
            # conf_loss_mask = 1 -> give penalty

            gt_bboxes = gt_bboxes[valid_mask,:]
            grid_tx = gt_bboxes[:,0] / self.stride
            grid_ty = gt_bboxes[:,1] / self.stride
            ti, tj = grid_tx.long().clamp(max=nW-1), grid_ty.long().clamp(max=nH-1)
            tn = best_n[valid_mask] # target anchor box number
            
            conf_loss_mask[b,tn,tj,ti] = 1
            gt_mask[b,tn,tj,ti] = 1
            tgt_xywh[b,tn,tj,ti,0] = grid_tx - grid_tx.floor()
            tgt_xywh[b,tn,tj,ti,1] = grid_ty - grid_ty.floor()
            tgt_xywh[b,tn,tj,ti,2] = torch.log(gt_bboxes[:,2]/self.anchors[tn,0] + 1e-8)
            tgt_xywh[b,tn,tj,ti,3] = torch.log(gt_bboxes[:,3]/self.anchors[tn,1] + 1e-8)
            tgt_conf[b,tn,tj,ti] = 1 # objectness confidence
            if self.n_cls > 0:
                tgt_cls[b, tn, tj, ti, gt_cls_idx[valid_mask]] = 1
            # smaller objects have higher losses
            img_area = img_size[0] * img_size[1]
            weighted[b,tn,tj,ti] = 2 - gt_bboxes[:,2] * gt_bboxes[:,3] / img_area

        # move the tagerts to GPU
        gt_mask = gt_mask.to(device=device)
        conf_loss_mask = conf_loss_mask.to(device=device)
        weighted = weighted.unsqueeze(-1).to(device=device)
        tgt_xywh = tgt_xywh.to(device=device)
        tgt_conf = tgt_conf.to(device=device)
        tgt_cls = tgt_cls.to(device=device)

        bce_logits = tnf.binary_cross_entropy_with_logits
        # weighted BCE loss for x,y
        loss_xy = bce_logits(t_xywh[...,0:2][gt_mask], tgt_xywh[...,0:2][gt_mask],
                             weight=weighted[gt_mask], reduction='sum')
        # weighted squared error for w,h
        loss_wh = (t_xywh[...,2:4][gt_mask] - tgt_xywh[...,2:4][gt_mask]).pow(2)
        loss_wh = 0.5*(weighted[gt_mask] * loss_wh).sum()
        loss_conf = bce_logits(conf_logits[conf_loss_mask],
                               tgt_conf[conf_loss_mask], reduction='sum')
        if self.n_cls > 0:
            loss_cls = bce_logits(cls_logits[gt_mask], tgt_cls[gt_mask], 
                                  reduction='sum')
        else:
            loss_cls = 0
        loss = loss_xy + loss_wh + loss_conf + loss_cls
        loss = loss / nB

        # logging
        ngt = valid_gt_num + 1e-16
        self.loss_str = f'yolo_{nH}x{nW} total {int(ngt)} objects: ' \
                        f'xy/gt {loss_xy/ngt:.3f}, wh/gt {loss_wh/ngt:.3f}, ' \
                        f'conf {loss_conf:.3f}, class {loss_cls:.3f}'
        self._assigned_num = valid_gt_num
        return preds, loss
