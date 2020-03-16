from time import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torchvision.transforms.functional as tvf

import models.backbones, models.fpns
from models.fcos import _xywh_to_ltrb, _ltrb_to_xywh
from models.losses import iou_loss
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
                                 stride=8)
        self.yolo_M = pred_layer(self.anchors_all, self.index_M, class_num,
                                 stride=16)
        self.yolo_L = pred_layer(self.anchors_all, self.index_L, class_num,
                                 stride=32)

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
        self.loss4conf = nn.BCEWithLogitsLoss(reduction='sum')

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
            less_ang: angle loss
            loss_conf: objectness loss - calculated by BCE.
            loss_l2: total l2 loss - only for logging.
        """
        assert raw.shape[2] == raw.shape[3]

        # raw shape(BatchSize, anchor_num*(5+cls_num), FeatureSize, FeatureSize)
        device = raw.device
        nB = raw.shape[0] # batch size
        nA = self.num_anchors # number of anchors
        nG = raw.shape[2] # grid size, i.e., prediction resolution
        nCH = 5 + self.n_cls # number of channels for each object
        assert nG * self.stride == img_size

        # tic = time()
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
        # training, convert final predictions to be normalized
        # preds[..., 0:4] /= img_size
        # preds[..., 0:4].clamp_(min=0, max=1)

        # tic = time()
        # nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        # assert (labels >= 0).all() and (labels[...,1:5] <= 1).all()

        # tcls_all = labels[:,:,0].long()
        # tx_all, ty_all = labels[:,:,1] * nG, labels[:,:,2] * nG # 0-nG
        # tw_all, th_all = labels[:,:,3], labels[:,:,4] # normalized 0-1

        # ti_all = tx_all.long()
        # tj_all = ty_all.long()

        # norm_anch_wh = self.anchors[:,0:2] / img_size # normalized
        # norm_anch_00wh = self.anch_00wh_all.clone()
        # norm_anch_00wh[:,2:4] /= img_size # normalized
        # torch.cuda.synchronize()
        # self.time_dic['process_label_anchor'] += time() - tic

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

            # gt_cls = im_labels[:,0]
            # gt_x, gt_y = im_labels[:,1], im_labels[:,2]
            # gt_w, gt_h = im_labels[:,3], im_labels[:,4]
            im_labels = im_labels[valid_mask,:]
            grid_tx = im_labels[:,1] / self.stride
            grid_ty = im_labels[:,2] / self.stride
            ti, tj = grid_tx.long(), grid_ty.long()
            tn = best_n[valid_mask]
            
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
            # weighted[b,tn,tj,ti] = torch.sqrt(2 - im_labels[:,3]*im_labels[:,4])
            weighted[b,tn,tj,ti] = 2 - im_labels[:,3]*im_labels[:,4]/img_size/img_size

        # tic = time()
        # move the tagerts to GPU
        gt_mask = gt_mask.to(device=device)
        conf_loss_mask = conf_loss_mask.to(device=device)
        weighted = weighted.unsqueeze(-1).to(device=device)
        target = target.to(device=device)

        # weighted BCE loss for x,y
        bce_logits = tnf.binary_cross_entropy_with_logits
        loss_xy = bce_logits(raw[...,0:2][gt_mask], target[...,0:2][gt_mask],
                             weight=weighted[gt_mask], reduction='sum')
        # weighted squared error for w,h
        loss_wh = (raw[...,2:4][gt_mask] - target[...,2:4][gt_mask]).pow(2)
        loss_wh = (weighted[gt_mask] * loss_wh).sum()
        # loss_wh = (wh_pred, wh_target, reduction='sum')
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
        self.anchors = self.anchors_all[anchor_indices]
        # anchors: tensor, e.g. shape(2,3), [[116,90],[156,198]]
        self.num_anchors = len(anchor_indices)
        # all anchors, (0, 0, w, h), used for calculating IoU
        self.anch_00wh_all = torch.zeros(len(self.anchors_all), 4)
        self.anch_00wh_all[:,2:] = self.anchors_all # absolute
        self.n_cls = class_num
        self.stride = kwargs['stride']

        self.ignore_thre = 0.6
        # self.loss4bbox = iou_loss
        self.loss4bbox = tnf.mse_loss
        # self.loss4bbox = tnf.l1_loss
        # self.loss4conf = FocalBCE(reduction='sum')
        self.loss4conf = nn.BCELoss(reduction='sum')

        self.time_dic = defaultdict(float)

    def forward(self, raw, img_size, labels=None):
        """
        Args:
        raw: input raw detections
        labels: label data whose size is :(N, K, 5)`. \
            N and K denote batchsize and number of labels. \
            Each label consists of [xc, yc, w, h, angle]: \
            xc, yc (float): center of bbox whose values range from 0 to 1. \
            w, h (float): size of bbox whose values range from 0 to 1. \
            angle (float): angle, degree from 0 to max_angle
        
        Returns:
        loss: total loss - the target of backprop.
        loss_
        loss_conf: objectness loss - calculated by BCE.
        """
        assert raw.shape[2] == raw.shape[3]
        assert img_size > 0 and isinstance(img_size, int)

        # raw shape(BatchSize, anchor_num*(5+cls_num), FeatureSize, FeatureSize)
        device = raw.device
        nB = raw.shape[0] # batch size
        nA = self.num_anchors # number of anchors
        nG = raw.shape[2] # grid size, or resolution
        nCH = 5 + self.n_cls # number of channels for each object
        assert nG * self.stride == img_size

        raw = raw.view(nB, nA, nCH, nG, nG)
        raw = raw.permute(0, 1, 3, 4, 2).contiguous()
        # now shape(nB, nA, nG, nG, nCH), meaning (nB x nA x nG x nG) predictions

        # ReLU activation for ltrb
        ltrb_raw = raw[..., 0:4]
        # ltrb_raw = tnf.relu_(raw[..., 0:4])
        # ltrb_raw = torch.exp(raw[..., 0:4])
        # logistic activation for objectness confidence
        conf = torch.sigmoid(raw[..., 4])
        if self.n_cls > 0:
            classes = torch.sigmoid(raw[..., 5:])

        # preds: (x,y,w,h,conf,cls...) in the grid normalized range
        preds = torch.empty(nB, nA, nG, nG, nCH, device=device)
        anch_w = self.anchors[:,0].view(1, nA, 1, 1, 1).cuda()
        ltrb_ = torch.exp(ltrb_raw.detach()) * anch_w
        preds[..., 0:4] = _ltrb_to_xywh(ltrb_ / self.stride, nG) * self.stride
        # confidence
        preds[..., 4] = conf.detach()
        # categories
        if self.n_cls > 0:
            preds[..., 5:] = classes.detach()
        preds = preds.cpu()
        # debug0 = angle[conf >= 0.1]
        # debug1 = preds[conf >= 0.1]

        if labels is None:
            return preds.view(nB, nA*nG*nG, nCH), None
        else:
            preds[..., 0:4] /= img_size
            # normalized between 0-1

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        assert (labels >= 0).all() and (labels[...,1:5] <= 1).all()

        tcls_all = labels[:,:,0].long()
        tx_all, ty_all = labels[:,:,1] * nG, labels[:,:,2] * nG # 0-nG
        tw_all, th_all = labels[:,:,3], labels[:,:,4] # normalized 0-1
        labels_ltrb = _xywh_to_ltrb(labels[:,:,1:5]*nG, nG).clamp(min=0) / nG

        ti_all = tx_all.long()
        tj_all = ty_all.long()

        norm_anch_00wh = self.anch_00wh_all.clone()
        norm_anch_00wh[:,2:4] /= img_size # normalized

        # traverse all images in a batch
        valid_gt_num = 0
        gt_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.bool)
        conf_loss_mask = torch.ones(nB, nA, nG, nG, dtype=torch.bool)
        # weighted = torch.zeros(nB, nA, nG, nG, 1)
        # regression and classification targets
        gt_ltrb = torch.zeros(nB, nA, nG, nG, 4)
        gt_conf = torch.zeros(nB, nA, nG, nG)
        if self.n_cls > 0:
            gt_cls = torch.zeros(nB, nA, nG, nG, self.n_cls)
        for b in range(nB):
            n = int(nlabel[b]) # number of ground truths in b'th image
            if n == 0:
                # no ground truth
                continue
            # assign gt to anchor box
            gt_00wh = torch.zeros(n, 4)
            gt_00wh[:, 2] = tw_all[b, :n] # normalized 0-1
            gt_00wh[:, 3] = th_all[b, :n] # normalized 0-1
            # calculate iou between truth and reference anchors
            anchor_ious = bboxes_iou(gt_00wh, norm_anch_00wh, xyxy=False)
            best_n_all = torch.argmax(anchor_ious, dim=1)
            best_n = best_n_all % self.num_anchors
            
            valid_mask = torch.zeros(n, dtype=torch.bool)
            for ind in self.anchor_indices:
                valid_mask = ( valid_mask | (best_n_all == ind) )
            if valid_mask.sum() == 0:
                # no anchor is responsible for any ground truth
                continue
            else:
                valid_gt_num += sum(valid_mask)

            gt_00wh[:, 0] = tx_all[b, :n] / nG # normalized 0-1
            gt_00wh[:, 1] = ty_all[b, :n] / nG # normalized 0-1

            # set conf_loss_mask to zero (ignore) if pred matches a gt more than thres
            ious = bboxes_iou(preds[b,...,0:4].view(-1,4), gt_00wh, xyxy=False)
            pred_best_iou, _ = ious.max(dim=1)
            ignore_mask = (pred_best_iou > self.ignore_thre)
            ignore_mask = ignore_mask.view(preds[b,...,0:4].shape[:3])
            conf_loss_mask[b] = ~ignore_mask
            # conf_loss_mask = 1 -> give penalty

            best_n = best_n[valid_mask]
            truth_i = ti_all[b, :n][valid_mask]
            truth_j = tj_all[b, :n][valid_mask]

            gt_mask[b,best_n,truth_j,truth_i] = 1
            conf_loss_mask[b,best_n,truth_j,truth_i] = 1
            ltrb_ = labels_ltrb[b,:n,:][valid_mask,:] # 0-1
            anch_w = self.anchors[best_n,0:1] / img_size # 0-1
            # anch_h = self.anchors[best_n,1] / self.stride # 0-nG
            gt_ltrb[b,best_n,truth_j,truth_i,:] = torch.log(ltrb_/anch_w + 1e-16)
            gt_conf[b,best_n,truth_j,truth_i] = 1 # objectness confidence
            if self.n_cls > 0:
                gt_cls[b,best_n,truth_j,truth_i,tcls_all[b,:n][valid_mask]] = 1

        # move the tagerts to GPU
        gt_mask = gt_mask.to(device=device)
        conf_loss_mask = conf_loss_mask.to(device=device)
        gt_ltrb = gt_ltrb.to(device=device)
        gt_conf = gt_conf.to(device=device)
        gt_cls = gt_cls.to(device=device)

        loss_bbox = self.loss4bbox(ltrb_raw[gt_mask], gt_ltrb[gt_mask],
                                   reduction='sum')
        loss_conf = self.loss4conf(conf[conf_loss_mask], gt_conf[conf_loss_mask])
        if self.n_cls > 0:
            loss_cls = tnf.binary_cross_entropy(classes[gt_mask], gt_cls[gt_mask],
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
