import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tnf

import models.losses as lossLib
from utils.bbox_ops import iou_rle


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
        # all anchors, rows of (0, 0, w, h, a). Used for calculating IoU
        self.anch_00wha_all = torch.zeros(len(anchors_all), 5)
        self.anch_00wha_all[:,2:4] = anchors_all # unnormalized

        self.ignore_thre = 0.6
        self.num_anchors = len(indices)
        self.stride = cfg['model.fpn.out_strides'][level_i]
        self.n_cls = cfg['general.num_class']

        loss_angle = cfg['model.angle.loss_angle']
        if loss_angle == 'period_L1':
            self.loss4angle = lossLib.period_L1(reduction='sum')
        elif loss_angle == 'period_L2':
            self.loss4angle = lossLib.period_L2(reduction='sum')
        else:
            raise Exception('unknown loss for angle')
        self.laname = loss_angle
        self.angle_range = cfg['model.angle.pred_range']
        assert self.angle_range in {180, 360}

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
        
        # Activation function only for angle. Leave other logits unchanged.
        # Convert the predicted angle to radian for later use.
        if self.laname in {'LL1', 'LL2'}:
            p_radian = t_xywha[..., 4]
        elif self.angle_range == 360:
            p_radian = torch.sigmoid(t_xywha[..., 4]) * 2 * np.pi - np.pi
        elif self.angle_range == 180:
            p_radian = torch.sigmoid(t_xywha[..., 4]) * np.pi - np.pi/2

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
        p_xywha = p_xywha.view(nB, nA*nH*nW, 5).cpu()
        
        if labels is None:
            # Logistic activation for confidence score
            p_conf = torch.sigmoid(conf_logits)
            # Logistic activation for categories
            if self.n_cls > 0:
                p_cls = torch.sigmoid(cls_logits)
                cls_score, cls_idx = torch.max(p_cls, dim=-1, keepdim=True)
                confs = p_conf * cls_score
                preds = {
                    'bbox': p_xywha,
                    'class_idx': cls_idx.view(nB, nA*nH*nW).cpu(),
                    'conf': confs.view(nB, nA*nH*nW).cpu(),
                }
            else:
                preds = {
                    'bbox': p_xywha,
                    'class_idx': torch.zeros(nB, nA*nH*nW, dtype=torch.int64),
                    'conf': p_conf.view(nB, nA*nH*nW).cpu(),
                }
            return preds, None

        assert isinstance(labels, list)
        valid_gt_num = 0
        # Set prediction targets
        gt_mask = torch.zeros(nB, nA, nH, nW, dtype=torch.bool)
        conf_loss_mask = torch.ones(nB, nA, nH, nW, dtype=torch.bool)
        weighted = torch.zeros(nB, nA, nH, nW)
        tgt_xywh = torch.zeros(nB, nA, nH, nW, 4)
        tgt_angle = torch.zeros(nB, nA, nH, nW, 1)
        tgt_conf = torch.zeros(nB, nA, nH, nW, 1)
        if self.n_cls > 0:
            tgt_cls = torch.zeros(nB, nA, nH, nW, self.n_cls)
        
        # traverse all images in a batch
        for b in range(nB):
            im_labels = labels[b]
            im_labels.sanity_check()
            nGT = len(im_labels)
            if nGT == 0:
                # no ground truth
                continue
            gt_bboxes = im_labels.bboxes
            gt_cls_idx = im_labels.cats
            assert gt_bboxes.shape[1] == 6 # (cls_idx, x, y, w, h, a)

            # calculate iou between truth and reference anchors
            _gt_00wh0 = torch.zeros(nGT, 5)
            _gt_00wh0[:, 2:4] = gt_bboxes[:, 2:4]
            _gt_00wh0[:, 4] = 0
            # anchor_ious = iou_mask(gt_boxes, norm_anch_00wha, xywha=True,
            #                        mask_size=64, is_degree=True)
            anchor_ious = iou_rle(_gt_00wh0, self.anch_00wha_all, xywha=True,
                                is_degree=True, img_size=img_size, normalized=False)
            best_n_all = torch.argmax(anchor_ious, dim=1)
            best_n = best_n_all % self.num_anchors
            
            valid_mask = torch.zeros(nGT, dtype=torch.bool)
            for ind in self.anchor_indices:
                valid_mask = ( valid_mask | (best_n_all == ind) )
            if valid_mask.sum() == 0:
                # no anchor is responsible for any ground truth
                continue
            else:
                valid_gt_num += sum(valid_mask)
            
            # best_n = best_n[valid_mask]
            # ti = ti_all[b, :n][valid_mask]
            # tj = tj_all[b, :n][valid_mask]

            # gt_boxes[:, 0] = tx_all[b, :n] / nG # normalized 0-1
            # gt_boxes[:, 1] = ty_all[b, :n] / nG # normalized 0-1
            # gt_boxes[:, 4] = ta_all[b, :n] # degree

            # logits > -7 <=> sigmoid(logits) > 0.0009
            selected_idx = conf_logits[b] > -7
            selected = p_xywha[b][selected_idx]
            if len(selected) < 2000 and len(selected) > 0:
                # ignore the predicted bboxes who have high overlap with any GT
                # pred_ious = iou_mask(selected.view(-1,5), gt_boxes, xywha=True,
                #                     mask_size=32, is_degree=True)
                pred_ious = iou_rle(selected.view(-1,5), im_labels[:,1:6], xywha=True,
                                is_degree=True, img_size=img_size, normalized=False)
                iou_with_gt, _ = pred_ious.max(dim=1)
                # ignore the conf of a pred BB if it matches a gt more than 0.7
                conf_loss_mask[b,selected_idx] = (iou_with_gt < self.ignore_thre)
                # conf_loss_mask = 1 -> give penalty
            
            im_labels = im_labels[valid_mask,:]
            grid_tx = im_labels[:,1] / self.stride
            grid_ty = im_labels[:,2] / self.stride
            ti, tj = grid_tx.long(), grid_ty.long()
            tn = best_n[valid_mask] # target anchor box number

            conf_loss_mask[b,tn,tj,ti] = 1
            gt_mask[b,tn,tj,ti] = 1
            tgt_xywh[b,tn,tj,ti,0] = grid_tx - grid_tx.floor()
            tgt_xywh[b,tn,tj,ti,1] = grid_ty - grid_ty.floor()
            tgt_xywh[b,tn,tj,ti,2] = torch.log(im_labels[:,3]/self.anchors[tn,0] + 1e-8)
            tgt_xywh[b,tn,tj,ti,3] = torch.log(im_labels[:,4]/self.anchors[tn,1] + 1e-8)
            # use radian when calculating loss
            tgt_angle[b,tn,tj,ti] = im_labels[:, 5] / 180 * np.pi
            tgt_conf[b,tn,tj,ti] = 1 # objectness confidence
            if self.n_cls > 0:
                tgt_cls[b, tn, tj, ti, im_labels[:,0].long()] = 1
            # smaller objects have higher losses
            img_area = img_size[0] * img_size[1]
            weighted[b,tn,tj,ti] = 2 - im_labels[:,3] * im_labels[:,4] / img_area

        # move the tagerts to GPU
        gt_mask = gt_mask.to(device=device)
        conf_loss_mask = conf_loss_mask.to(device=device)
        weighted = weighted.unsqueeze(-1).to(device=device)
        tgt_xywh = tgt_xywh.to(device=device)
        tgt_angle = tgt_angle.to(device=device)
        tgt_conf = tgt_conf.to(device=device)
        tgt_cls = tgt_cls.to(device=device)
        
        bce_logits = tnf.binary_cross_entropy_with_logits
        loss_xy = self.bce_loss(t_xywha[...,0:2][gt_mask], tgt_xywh[...,0:2][gt_mask],
                                weight=weighted[gt_mask], reduction='sum')
        # weighted squared error for w,h
        loss_wh = (t_xywha[...,2:4][gt_mask] - tgt_xywh[...,2:4][gt_mask]).pow(2)
        loss_wh = (weighted[gt_mask] * loss_wh).sum()
        # loss for angle
        loss_angle = self.loss4angle(p_radian[gt_mask], tgt_angle[gt_mask])
        loss_obj = bce_logits(conf_logits[conf_loss_mask],
                              tgt_conf[conf_loss_mask], reduction='sum')
        if self.n_cls > 0:
            loss_cls = bce_logits(cls_logits[gt_mask], tgt_cls[gt_mask], 
                                  reduction='sum')
        else:
            loss_cls = 0
        loss = loss_xy + 0.5*loss_wh + loss_angle + loss_obj

        # logging
        ngt = valid_gt_num + 1e-16
        self.loss_str = f'yolo_{nH}x{nW} total {int(ngt)} objects: ' \
                        f'xy/gt {loss_xy/ngt:.3f}, wh/gt {loss_wh/ngt:.3f}, ' \
                        f'angle/gt {loss_angle/ngt:.3f}, conf {loss_obj:.3f}, ' \
                        f'class {loss_cls:.3f}'
        self.gt_num = valid_gt_num
        return None, loss
