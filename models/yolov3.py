from time import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torchvision.transforms.functional as tvf

import models.backbones, models.fpns
import models.losses as lossLib
from models.fcos import _xywh_to_ltrb, _ltrb_to_xywh
from utils.iou_funcs import bboxes_iou
from utils.structures import ImageObjects
# from utils.timer import contexttimer


class YOLOv3(nn.Module):
    def __init__(self, class_num=80, backbone='dark53', **kwargs):
        super().__init__()
        self.input_normalization = kwargs.get('img_norm', False)
        anchors = [
            [10, 13], [16, 30], [33, 23],
            [30, 61], [62, 45], [59, 119],
            [116, 90], [156, 198], [373, 326]
        ]
        indices = [[6,7,8], [3,4,5], [0,1,2]]
        self.anchors_all = torch.Tensor(anchors).float()
        assert self.anchors_all.shape[1] == 2 and len(indices) == 3
        self.index_L = torch.Tensor(indices[0]).long()
        self.index_M = torch.Tensor(indices[1]).long()
        self.index_S = torch.Tensor(indices[2]).long()

        if backbone == 'dark53':
            self.backbone = models.backbones.Darknet53()
            chs = (256, 512, 1024)
            print("Using backbone Darknet-53. Loading ImageNet weights....")
            pretrained = torch.load('./weights/dark53_imgnet.pth')
            self.load_state_dict(pretrained)
        elif backbone == 'res34':
            self.backbone = models.backbones.resnet34()
        elif backbone == 'res50':
            self.backbone = models.backbones.resnet50()
        elif backbone == 'res101':
            self.backbone = models.backbones.resnet101()
        elif 'efficientnet' in backbone:
            self.backbone = models.backbones.efficientnet(backbone)
        else:
            raise Exception('Unknown backbone name')

        self.fpn = models.fpns.YOLOv3FPN(in_channels=chs, class_num=class_num)
        
        pred_layer_name = kwargs.get('pred_layer', 'YOLO')
        if pred_layer_name == 'YOLO':
            pred_layer = YOLOLayer
        elif pred_layer_name == 'FCOS':
            pred_layer = FCOSLayer
        self.yolo_S = pred_layer(self.anchors_all, self.index_S, class_num,
                                 stride=8, **kwargs)
        self.yolo_M = pred_layer(self.anchors_all, self.index_M, class_num,
                                 stride=16, **kwargs)
        self.yolo_L = pred_layer(self.anchors_all, self.index_L, class_num,
                                 stride=32, **kwargs)

        self.time_dic = defaultdict(float)

    def forward(self, x, labels=None):
        '''
        x: a batch of images, e.g. shape(8,3,608,608)
        labels: a batch of ground truth
        '''
        torch.cuda.reset_max_memory_allocated()
        # assert x.dim() == 4 and x.shape[2] == x.shape[3]
        # assert ((x>=-0.5) & (x<=1.5)).all()
        self.img_size = x.shape[2]
        # normalization
        if self.input_normalization:
            for i in range(x.shape[0]):
                x[i] = tvf.normalize(x[i], [0.485,0.456,0.406], [0.229,0.224,0.225],
                                    inplace=True)
                # debug = (x.mean(), x.std())

        # go through the backbone and the feature payamid network
        # tic = time()
        features = self.backbone(x)
        # torch.cuda.synchronize()
        # self.time_dic['backbone'] += time() - tic

        # tic = time()
        features_fpn = self.fpn(features)
        # torch.cuda.synchronize()
        # self.time_dic['fpn'] += time() - tic

        # process the boxes, and calculate loss if there is gt
        p3_fpn, p4_fpn, p5_fpn = features_fpn
        # tic = time()
        boxes_S, loss_S = self.yolo_S(p3_fpn, self.img_size, labels)
        # torch.cuda.synchronize()
        # self.time_dic['head_1'] += time() - tic

        # tic = time()
        boxes_M, loss_M = self.yolo_M(p4_fpn, self.img_size, labels)
        # torch.cuda.synchronize()
        # self.time_dic['head_2'] += time() - tic
        
        # tic = time()
        boxes_L, loss_L = self.yolo_L(p5_fpn, self.img_size, labels)
        # torch.cuda.synchronize()
        # self.time_dic['head_3'] += time() - tic

        if labels is None:
            # tic = time()
            # assert boxes_L.dim() == 3
            boxes = torch.cat((boxes_L,boxes_M,boxes_S), dim=1)
            bbs, confs, clss = boxes[...,:4], boxes[...,4], boxes[...,5:]
            cls_score, cls_idx = clss.max(dim=2, keepdim=False)
            # boxes = torch.cat([cls_idx.float(),boxes[...,0:5]], dim=2)
            # boxes[:,:,4] *= cls_score.squeeze(-1)
            # debug = boxes[boxes[...,5]>0.5]
            # self.time_dic['cat_box'] += time() - tic
            bb = ImageObjects(bboxes=bbs, cats=cls_idx, scores=confs*cls_score)
            return bb
        else:
            # check all the gt objects are assigned
            assert isinstance(labels, list)
            gt_num = sum([t.shape[0] for t in labels])
            assigned = self.yolo_L.gt_num + self.yolo_M.gt_num + self.yolo_S.gt_num
            assert assigned == gt_num
            self.loss_str = self.yolo_L.loss_str + '\n' + self.yolo_M.loss_str + \
                            '\n' + self.yolo_S.loss_str
            loss = loss_L + loss_M + loss_S
            return loss
    
    def time_monitor(self):
        s = str(self.time_dic)
        # s += '\nbranch_1:' + str(self.yolo_S.time_dic)
        # s += '\nbranch_2:' + str(self.yolo_M.time_dic)
        # s += '\nbranch_3:' + str(self.yolo_L.time_dic)
        return s


class YOLOLayer(nn.Module):
    '''
    calculate the output boxes and losses
    '''
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
        # self.loss4conf = FocalBCE(reduction='sum')
        # self.loss4conf = nn.BCEWithLogitsLoss(reduction='sum')

        self.time_dic = defaultdict(float)

    def forward(self, raw, img_size, labels=None):
        """
        Args:
            raw: input raw detections
            labels: list[tensor]`. \
                N and K denote batchsize and number of labels. \
                Each label consists of [xc, yc, w, h, angle]: \
                xc, yc (float): center of bbox whose values range from 0 to 1. \
                w, h (float): size of bbox whose values range from 0 to 1. \
                angle (float): angle, degree from 0 to max_angle
        
        Returns:
            loss: total loss - the target of backprop.
            loss_xy: x, y loss - calculated by binary cross entropy (BCE) \
                with boxsize-dependent weights.
            loss_wh: w, h loss - calculated by l2 without averaging and \
                with boxsize-dependent weights.
            loss_conf: objectness loss - calculated by BCE.
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
        # (x,y,w,h,conf,categories)

        # ----------------------- logits to prediction -----------------------
        preds = raw.detach().clone()
        # x, y
        x_ = torch.arange(nG, dtype=torch.float, device=device)
        mesh_y, mesh_x = torch.meshgrid(x_, x_)
        preds[..., 0] = (torch.sigmoid(preds[..., 0]) + mesh_x) * self.stride
        preds[..., 1] = (torch.sigmoid(preds[..., 1]) + mesh_y) * self.stride
        # w, h
        anch_wh = self.anchors.view(1,nA,1,1,2).to(device=device)
        preds[...,2:4] = torch.exp(preds[...,2:4]) * anch_wh
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

            pred_ious = bboxes_iou(preds[b,:,0:4], im_labels[:,1:5], xyxy=False)
            iou_with_gt, _ = pred_ious.max(dim=1)
            # ignore the conf of a pred BB if it matches a gt more than 0.7
            conf_loss_mask[b] = (iou_with_gt < self.ignore_thre).view(nA,nG,nG)
            # conf_loss_mask = 1 -> give penalty

            im_labels = im_labels[valid_mask,:]
            grid_tx = im_labels[:,1] / self.stride
            grid_ty = im_labels[:,2] / self.stride
            ti, tj = grid_tx.long(), grid_ty.long()
            tn = best_n[valid_mask] # target anchor box number
            
            conf_loss_mask[b,tn,tj,ti] = 1
            gt_mask[b,tn,tj,ti] = 1
            target[b,tn,tj,ti,0] = grid_tx - grid_tx.floor()
            target[b,tn,tj,ti,1] = grid_ty - grid_ty.floor()
            target[b,tn,tj,ti,2] = torch.log(im_labels[:,3]/self.anchors[tn,0] + 1e-8)
            target[b,tn,tj,ti,3] = torch.log(im_labels[:,4]/self.anchors[tn,1] + 1e-8)
            target[b,tn,tj,ti,4] = 1 # objectness confidence
            if self.n_cls > 0:
                target[b, tn, tj, ti, 5 + im_labels[:,0].long()] = 1
            # smaller objects have higher losses
            weighted[b,tn,tj,ti] = 2 - im_labels[:,3]*im_labels[:,4]/img_size/img_size

        # move the tagerts to GPU
        gt_mask = gt_mask.to(device=device)
        conf_loss_mask = conf_loss_mask.to(device=device)
        weighted = weighted.unsqueeze(-1).to(device=device)
        target = target.to(device=device)

        bce_logits = tnf.binary_cross_entropy_with_logits
        # weighted BCE loss for x,y
        loss_xy = bce_logits(raw[...,0:2][gt_mask], target[...,0:2][gt_mask],
                             weight=weighted[gt_mask], reduction='sum')
        # weighted squared error for w,h
        loss_wh = (raw[...,2:4][gt_mask] - target[...,2:4][gt_mask]).pow(2)
        loss_wh = (weighted[gt_mask] * loss_wh).sum()
        loss_conf = bce_logits(raw[...,4][conf_loss_mask],
                               target[...,4][conf_loss_mask], reduction='sum')
        if self.n_cls > 0:
            loss_cls = bce_logits(raw[...,5:][gt_mask], target[...,5:][gt_mask], 
                                  reduction='sum')
        else:
            loss_cls = 0
        loss = loss_xy + 0.5*loss_wh + loss_conf + loss_cls

        # logging
        ngt = valid_gt_num + 1e-16
        self.loss_str = f'yolo_{nG} total {int(ngt)} objects: ' \
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

        self.time_dic = defaultdict(float)

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
        # (x,y,w,h,conf,categories)
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
