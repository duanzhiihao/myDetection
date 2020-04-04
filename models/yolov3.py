import torch
import torch.nn as nn
import torch.nn.functional as tnf

from .backbones import get_backbone_fpn
from .rpns import get_rpn
import models.losses as lossLib
from models.fcos import _xywh_to_ltrb, _ltrb_to_xywh
from utils.iou_funcs import bboxes_iou
from utils.structures import ImageObjects


class YOLOv3(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone, self.fpn, fpn_info = get_backbone_fpn(cfg['backbone_fpn'])
        self.rpn = get_rpn(cfg['rpn'], fpn_info['feature_channels'], **cfg)
        
        if cfg['pred_layer'] == 'YOLO':
            pred_layer = YOLOLayer
        elif cfg['pred_layer'] == 'FCOS':
            pred_layer = FCOSLayer
        self.bb_layers = nn.ModuleList()
        strides = fpn_info['feature_strides']
        for level_i, s in enumerate(strides):
            self.bb_layers.append(pred_layer(strides_all=strides, stride=s,
                                             level=level_i, **cfg))

        self.input_format = cfg['input_format']

    def forward(self, x, labels=None):
        '''
        x: a batch of images, e.g. shape(8,3,608,608)
        labels: a batch of ground truth
        '''
        assert x.dim() == 4
        self.img_size = x.shape[2:4]

        # go through the backbone and the feature payamid network
        features = self.backbone(x)
        features = self.fpn(features)
        all_branch_preds = self.rpn(features)
        
        dts_all = []
        losses_all = []
        for i, raw_preds in enumerate(all_branch_preds):
            dts, loss = self.bb_layers[i](raw_preds, self.img_size, labels)
            dts_all.append(dts)
            losses_all.append(loss)

        if labels is None:
            batch_bbs = torch.cat([d['bbox'] for d in dts_all], dim=1)
            batch_cls_idx = torch.cat([d['class_idx'] for d in dts_all], dim=1)
            batch_confs = torch.cat([d['conf'] for d in dts_all], dim=1)

            p_objects = []
            for bbs, cls_idx, confs in zip(batch_bbs, batch_cls_idx, batch_confs):
                p_objects.append(ImageObjects(bboxes=bbs, cats=cls_idx, scores=confs))
            return p_objects
        else:
            assert isinstance(labels, list)
            # total_gt_num = sum([t.shape[0] for t in labels])
            # assigned_gt_num = sum(branch._assigned_num for branch in self.bb_layers)
            self.loss_str = ''
            for m in self.bb_layers:
                self.loss_str += m.loss_str + '\n'
            loss = sum(losses_all)
            return loss


class YOLOLayer(nn.Module):
    '''
    calculate the output boxes and losses
    '''
    def __init__(self, **kwargs):
        super().__init__()
        anchors = [
            [10, 13], [16, 30], [33, 23],
            [30, 61], [62, 45], [59, 119],
            [116, 90], [156, 198], [373, 326]
        ]
        indices = [[0,1,2], [3,4,5], [6,7,8]]
        self.anchors_all = torch.Tensor(anchors).float()

        level_i = kwargs['level']
        self.stride = kwargs['strides_all'][level_i]
        self.n_cls = kwargs['num_class']

        anchor_indices = torch.Tensor(indices[level_i]).long()
        self.anchor_indices = anchor_indices
        self.anchors = self.anchors_all[anchor_indices, :]
        # anchors: tensor, e.g. shape(3,2), [[116, 90], [156, 198], [373, 326]]
        self.num_anchors = len(anchor_indices)
        # all anchors, (0, 0, w, h), used for calculating IoU
        self.anch_00wh_all = torch.zeros(len(self.anchors_all), 4)
        self.anch_00wh_all[:,2:] = self.anchors_all # unnormalized

        self.ignore_thre = 0.6

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
        p_xywh = t_xywh.detach().clone()
        # x, y
        y_ = torch.arange(nH, dtype=torch.float, device=device)
        x_ = torch.arange(nW, dtype=torch.float, device=device)
        mesh_y, mesh_x = torch.meshgrid(y_, x_)
        p_xywh[..., 0] = (torch.sigmoid(p_xywh[..., 0]) + mesh_x) * self.stride
        p_xywh[..., 1] = (torch.sigmoid(p_xywh[..., 1]) + mesh_y) * self.stride
        # w, h
        anch_wh = self.anchors.view(1,nA,1,1,2).to(device=device)
        p_xywh[...,2:4] = torch.exp(p_xywh[...,2:4]) * anch_wh
        p_xywh = p_xywh.view(nB, nA*nH*nW, 4).cpu()

        if labels is None:
            # Logistic activation for confidence score
            p_conf = torch.sigmoid(conf_logits)
            # Logistic activation for categories
            if self.n_cls > 0:
                p_cls = torch.sigmoid(cls_logits)
            cls_score, cls_idx = torch.max(p_cls, dim=-1, keepdim=True)
            confs = p_conf * cls_score
            preds = {
                'bbox': p_xywh,
                'class_idx': cls_idx.view(nB, nA*nH*nW).cpu(),
                'conf': confs.view(nB, nA*nH*nW).cpu(),
            }
            return preds, None
            
        assert isinstance(labels, list)
        # traverse all images in a batch
        valid_gt_num = 0
        gt_mask = torch.zeros(nB, nA, nH, nW, dtype=torch.bool)
        conf_loss_mask = torch.ones(nB, nA, nH, nW, dtype=torch.bool)
        weighted = torch.zeros(nB, nA, nH, nW)
        tgt_xywh = torch.zeros(nB, nA, nH, nW, 4)
        tgt_conf = torch.zeros(nB, nA, nH, nW, 1)
        tgt_cls = torch.zeros(nB, nA, nH, nW, self.n_cls)
        for b in range(nB):
            im_labels = labels[b]
            num_gt = im_labels.shape[0]
            if num_gt == 0:
                # no ground truth
                continue
            assert im_labels.shape[1] == 5

            # calculate iou between truth and reference anchors
            gt_00wh = torch.zeros(num_gt, 4)
            gt_00wh[:, 2:4] = im_labels[:, 3:5]
            anchor_ious = bboxes_iou(gt_00wh, self.anch_00wh_all, xyxy=False)
            best_n_all = torch.argmax(anchor_ious, dim=1)
            best_n = best_n_all % self.num_anchors
            
            valid_mask = torch.zeros(num_gt, dtype=torch.bool)
            for ind in self.anchor_indices:
                valid_mask = ( valid_mask | (best_n_all == ind) )
            if valid_mask.sum() == 0:
                # no anchor is responsible for any ground truth
                continue
            else:
                valid_gt_num += sum(valid_mask)

            pred_ious = bboxes_iou(p_xywh[b], im_labels[:,1:5], xyxy=False)
            iou_with_gt, _ = pred_ious.max(dim=1)
            # ignore the conf of a pred BB if it matches a gt more than 0.7
            conf_loss_mask[b] = (iou_with_gt < self.ignore_thre).view(nA,nH,nW)
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
        tgt_conf = tgt_conf.to(device=device)
        tgt_cls = tgt_cls.to(device=device)

        bce_logits = tnf.binary_cross_entropy_with_logits
        # weighted BCE loss for x,y
        loss_xy = bce_logits(t_xywh[...,0:2][gt_mask], tgt_xywh[...,0:2][gt_mask],
                             weight=weighted[gt_mask], reduction='sum')
        # weighted squared error for w,h
        loss_wh = (t_xywh[...,2:4][gt_mask] - tgt_xywh[...,2:4][gt_mask]).pow(2)
        loss_wh = (weighted[gt_mask] * loss_wh).sum()
        loss_conf = bce_logits(conf_logits[conf_loss_mask],
                               tgt_conf[conf_loss_mask], reduction='sum')
        if self.n_cls > 0:
            loss_cls = bce_logits(cls_logits[gt_mask], tgt_cls[gt_mask], 
                                  reduction='sum')
        else:
            loss_cls = 0
        loss = loss_xy + 0.5*loss_wh + loss_conf + loss_cls

        # logging
        ngt = valid_gt_num + 1e-16
        self.loss_str = f'yolo_{nH}x{nW} total {int(ngt)} objects: ' \
                        f'xy/gt {loss_xy/ngt:.3f}, wh/gt {loss_wh/ngt:.3f}' \
                        f', conf {loss_conf:.3f}, class {loss_cls:.3f}'
        self.gt_num = valid_gt_num
        return None, loss


class FCOSLayer(nn.Module):
    def __init__(self, all_anchors, anchor_indices, class_num, **kwargs):
        super().__init__()
        self.anchors_all = all_anchors
        self.anchor_indices = anchor_indices
        self.anchors = self.anchors_all[anchor_indices, :]
        # anchors: tensor, e.g. shape(3,2), [[116, 90], [156, 198], [373, 326]]
        self.num_anchors = len(anchor_indices)
        # all anchors, (0, 0, w, h), used for calculating IoU
        self.anch_00wh_all = torch.zeros(len(self.anchors_all), 4)
        self.anch_00wh_all[:,2:] = self.anchors_all # unnormalized
        self.n_cls = class_num
        self.stride = kwargs['stride']

        self.ignore_thre = 0.6
        self.ltrb_setting = kwargs.get('ltrb', 'exp_l2')

    def forward(self, raw, img_size, labels=None):
        """
        Args:
            raw: input raw detections
            labels: list[tensor]
        
        Returns:
            loss: total loss
        """
        assert raw.shape[2] == raw.shape[3]

        # raw shape(BatchSize, anchor_num*(5+cls_num), FeatureSize, FeatureSize)
        device = raw.device
        nB = raw.shape[0] # batch size
        nA = self.num_anchors # number of anchors
        nG = raw.shape[2] # grid size, i.e., prediction resolution
        nCH = 5 + self.n_cls # number of channels for each object
        assert nG * self.stride == img_size

        raw = raw.view(nB, nA, nCH, nG, nG)
        raw = raw.permute(0, 1, 3, 4, 2).contiguous()
        # Now raw.shape is (nB, nA, nG, nG, nCH), where the last demension is
        # (l,t,r,b,conf,categories)
        if self.ltrb_setting.startswith('relu'):
            # ReLU activation
            tnf.relu(raw[..., 0:4], inplace=True)

        # ----------------------- logits to prediction -----------------------
        preds = raw.detach().clone()
        # left, top, right, bottom
        anch_w = self.anchors[:,0].view(1,nA,1,1).to(device=device)
        anch_h = self.anchors[:,1].view(1,nA,1,1).to(device=device)
        if self.ltrb_setting.startswith('exp'):
            preds[...,0] = torch.exp(preds[...,0]) * anch_w # unnormalized
            preds[...,1] = torch.exp(preds[...,1]) * anch_h # unnormalized
            preds[...,2] = torch.exp(preds[...,2]) * anch_w # unnormalized
            preds[...,3] = torch.exp(preds[...,3]) * anch_h # unnormalized
        elif self.ltrb_setting.startswith('relu'):
            preds[...,0] = preds[...,0] * anch_w # unnormalized
            preds[...,1] = preds[...,1] * anch_h # unnormalized
            preds[...,2] = preds[...,2] * anch_w # unnormalized
            preds[...,3] = preds[...,3] * anch_h # unnormalized
        else:
            raise Exception('Unknown ltrb setting')
        preds[..., 0:4].clamp_(min=0, max=img_size)
        preds[..., 0:4] = _ltrb_to_xywh(preds[..., 0:4], nG, self.stride) # xywh
        # confidence
        preds[..., 4] = torch.sigmoid(preds[..., 4])
        # categories
        if self.n_cls > 0:
            preds[..., 5:] = torch.sigmoid(preds[..., 5:])
        preds = preds.view(nB, nA*nG*nG, nCH).cpu()
        # debug0 = angle[conf >= 0.1]
        # debug1 = preds[conf >= 0.1]

        if labels is None:
            return preds, None
            
        assert isinstance(labels, list)
        # traverse all images in a batch
        valid_gt_num = 0
        gt_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.bool)
        conf_loss_mask = torch.ones(nB, nA, nG, nG, dtype=torch.bool)
        weighted = torch.zeros(nB, nA, nG, nG)
        target = torch.zeros(nB, nA, nG, nG, nCH)
        for b in range(nB):
            num_gt = labels[b].shape[0]
            if num_gt == 0:
                # no ground truth
                continue
            assert labels[b].shape[1] == 5
            gt_xywh = labels[b][:,1:5]

            # calculate iou between truth and reference anchors
            gt_00wh = torch.zeros(num_gt, 4)
            gt_00wh[:, 2:4] = gt_xywh[:, 2:4]
            anchor_ious = bboxes_iou(gt_00wh, self.anch_00wh_all, xyxy=False)
            best_n_all = torch.argmax(anchor_ious, dim=1)
            best_n = best_n_all % self.num_anchors
            
            valid_mask = torch.zeros(num_gt, dtype=torch.bool)
            for ind in self.anchor_indices:
                valid_mask = ( valid_mask | (best_n_all == ind) )
            if valid_mask.sum() == 0:
                # no anchor is responsible for any ground truth
                continue
            else:
                valid_gt_num += sum(valid_mask)

            pred_ious = bboxes_iou(preds[b,:,0:4], gt_xywh, xyxy=False)
            iou_with_gt, _ = pred_ious.max(dim=1)
            # ignore the conf of a pred BB if it matches a gt more than 0.7
            conf_loss_mask[b] = (iou_with_gt < self.ignore_thre).view(nA,nG,nG)
            # conf_loss_mask = 1 -> give penalty

            gt_xywh = gt_xywh[valid_mask,:]
            gt_ltrb = _xywh_to_ltrb(gt_xywh, nG, self.stride)
            gt_ltrb.clamp_(min=0)
            gt_cls = labels[b][valid_mask,0].long()

            grid_tx = gt_xywh[:,0] / self.stride
            grid_ty = gt_xywh[:,1] / self.stride
            ti, tj = grid_tx.long(), grid_ty.long()
            tn = best_n[valid_mask] # target anchor box number
            anch_w = self.anchors[tn,0]
            anch_h = self.anchors[tn,1]
            
            conf_loss_mask[b,tn,tj,ti] = 1
            gt_mask[b,tn,tj,ti] = 1
            if self.ltrb_setting.startswith('exp'):
                target[b,tn,tj,ti,0] = torch.log(gt_ltrb[:,0] / anch_w + 1e-8)
                target[b,tn,tj,ti,1] = torch.log(gt_ltrb[:,1] / anch_h + 1e-8)
                target[b,tn,tj,ti,2] = torch.log(gt_ltrb[:,2] / anch_w + 1e-8)
                target[b,tn,tj,ti,3] = torch.log(gt_ltrb[:,3] / anch_h + 1e-8)
            elif self.ltrb_setting.startswith('relu'):
                target[b,tn,tj,ti,0] = gt_ltrb[:,0] / anch_w
                target[b,tn,tj,ti,1] = gt_ltrb[:,1] / anch_h
                target[b,tn,tj,ti,2] = gt_ltrb[:,2] / anch_w
                target[b,tn,tj,ti,3] = gt_ltrb[:,3] / anch_h
            target[b,tn,tj,ti,4] = 1 # objectness confidence
            if self.n_cls > 0:
                target[b, tn, tj, ti, 5 + gt_cls] = 1
            # smaller objects have higher losses
            weighted[b,tn,tj,ti] = 2 - gt_xywh[:,2]*gt_xywh[:,3]/img_size/img_size

        # move the tagerts to GPU
        gt_mask = gt_mask.to(device=device)
        conf_loss_mask = conf_loss_mask.to(device=device)
        weighted = weighted.unsqueeze(-1).to(device=device)[gt_mask]
        target = target.to(device=device)

        bce_logits = tnf.binary_cross_entropy_with_logits
        p_ltrb, target_ltrb = raw[...,0:4][gt_mask], target[...,0:4][gt_mask]
        if self.ltrb_setting.endswith('l2'):
            # weighted squared error for l,t,r,b
            loss_bbox = (p_ltrb - target_ltrb).pow(2)
            loss_bbox = 0.5 * (weighted * loss_bbox).sum()
        elif self.ltrb_setting.endswith('sl1'):
            # weighted smooth L1 loss for l,t,r,b
            loss_bbox = lossLib.smooth_L1_loss(p_ltrb, target_ltrb, beta=1, 
                                               weight=weighted, reduction='sum')
        elif self.ltrb_setting.endswith('giou'):
            loss_bbox = lossLib.iou_loss(p_ltrb, target_ltrb, iou_type='giou', 
                                         reduction='sum')
        else:
            raise Exception('Unknown loss')
        loss_conf = bce_logits(raw[...,4][conf_loss_mask],
                               target[...,4][conf_loss_mask], reduction='sum')
        if self.n_cls > 0:
            loss_cls = bce_logits(raw[...,5:][gt_mask], target[...,5:][gt_mask], 
                                  reduction='sum')
        else:
            loss_cls = 0
        loss = loss_bbox + loss_conf + loss_cls

        # logging
        ngt = valid_gt_num + 1e-16
        self.loss_str = f'yolo_{nG} total {int(ngt)} objects: ' \
                        f'bbox/gt {loss_bbox/ngt:.3f}, ' \
                        f'conf {loss_conf:.3f}, class {loss_cls:.3f}'
        self.gt_num = valid_gt_num
        return None, loss
