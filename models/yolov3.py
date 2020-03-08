from time import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torchvision.transforms.functional as tvf

import models.backbones, models.fpns, models.losses
from utils.iou_funcs import bboxes_iou
# from utils.timer import contexttimer


class YOLOv3(nn.Module):
    def __init__(self, class_num=80, backbone='dark53', img_norm=False):
        super().__init__()
        self.input_normalization = img_norm
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
        
        self.yolo_S = YOLOLayer(self.anchors_all, self.index_S, class_num)
        self.yolo_M = YOLOLayer(self.anchors_all, self.index_M, class_num)
        self.yolo_L = YOLOLayer(self.anchors_all, self.index_L, class_num)

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
            cls_score, cls_idx = boxes[...,5:].max(dim=2, keepdim=True)
            boxes = torch.cat([cls_idx.float(),boxes[...,0:5]], dim=2)
            boxes[:,:,4] *= cls_score.squeeze(-1)
            # debug = boxes[boxes[...,5]>0.5]
            # self.time_dic['cat_box'] += time() - tic
            return boxes
        else:
            # check all the gt objects are assigned
            gt_num = (labels[:,:,0:4].sum(dim=2) > 0).sum()
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
    def __init__(self, all_anchors, anchor_indices, class_num):
        super().__init__()
        self.anchors_all = all_anchors
        self.anchor_indices = anchor_indices
        self.anchors = self.anchors_all[anchor_indices].cuda()
        # anchors: tensor, e.g. shape(2,3), [[116,90],[156,198]]
        self.num_anchors = len(anchor_indices)
        # all anchors, (0, 0, w, h), used for calculating IoU
        self.anch_00wh_all = torch.zeros(len(self.anchors_all), 4)
        self.anch_00wh_all[:,2:] = self.anchors_all # absolute
        self.n_cls = class_num

        self.ignore_thre = 0.6
        # self.loss4obj = FocalBCE(reduction='sum')
        self.loss4obj = nn.BCELoss(reduction='sum')

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
        loss_xy: x, y loss - calculated by binary cross entropy (BCE) \
            with boxsize-dependent weights.
        loss_wh: w, h loss - calculated by l2 without averaging and \
            with boxsize-dependent weights.
        less_ang: angle loss
        loss_obj: objectness loss - calculated by BCE.
        loss_l2: total l2 loss - only for logging.
        """
        assert raw.shape[2] == raw.shape[3]
        assert img_size > 0 and isinstance(img_size, int)

        # raw shape(BatchSize, anchor_num*(5+cls_num), FeatureSize, FeatureSize)
        device = raw.device
        nB = raw.shape[0] # batch size
        nA = self.num_anchors # number of anchors
        nG = raw.shape[2] # grid size, or resolution
        nCH = 5 + self.n_cls # number of channels for each object

        # tic = time()
        raw = raw.view(nB, nA, nCH, nG, nG)
        raw = raw.permute(0, 1, 3, 4, 2).contiguous()
        # now shape(nB, nA, nG, nG, nCH), meaning (nB x nA x nG x nG) predictions

        # logistic activation for xy, angle, obj_conf
        xy_offset = torch.sigmoid(raw[..., 0:2])
        # linear activation for w, h
        wh_scale = raw[..., 2:4]
        # now xy are the offsets, wh are the scaling factors of anchor boxes
        # logistic activation for objectness confidence
        conf = torch.sigmoid(raw[..., 4])
        if self.n_cls > 0:
            classes = torch.sigmoid(raw[..., 5:])
        # torch.cuda.synchronize()
        # self.time_dic['activations'] += time() - tic

        # tic = time()
        # calculate predicted - (x,y,w,h,conf,categories)
        x_shift = torch.arange(nG, dtype=torch.float,
                               device=device).repeat(nG,1).view(1,1,nG,nG)
        y_shift = torch.arange(nG, dtype=torch.float,
                               device=device).repeat(nG,1).t().view(1,1,nG,nG)
        # torch.cuda.synchronize()
        # self.time_dic['xy_arange'] += time() - tic
        # note that the anchors are not normalized
        anchors = self.anchors.clone()
        anch_w = anchors[:,0].view(1, nA, 1, 1) # absolute
        anch_h = anchors[:,1].view(1, nA, 1, 1) # absolute

        # tic = time()
        # these operations are performed on CPU in order to speed up
        txy = xy_offset.detach()
        twh = wh_scale.detach()
        pred_final = torch.empty(nB, nA, nG, nG, nCH, device=device)
        pred_final[..., 0] = (txy[..., 0] + x_shift) / nG # normalized 0-1
        pred_final[..., 1] = (txy[..., 1] + y_shift) / nG # normalized 0-1
        pred_final[..., 2] = torch.exp(twh[..., 0]) * anch_w # absolute
        pred_final[..., 3] = torch.exp(twh[..., 1]) * anch_h # absolute
        # confidence
        pred_final[..., 4] = conf.detach()
        # categories
        if self.n_cls > 0:
            pred_final[..., 5:] = classes.detach()
        pred_final = pred_final.cpu()
        # debug0 = angle[conf >= 0.1]
        # debug1 = pred_final[conf >= 0.1]
        # torch.cuda.synchronize()
        # self.time_dic['raw_to_bb'] += time() - tic

        if labels is None:
            # inference, convert final predictions to absolute
            pred_final[..., :2] *= img_size
            # debug = pred_final.view(nB, nA*nG*nG, nCH)
            # debug = debug[debug[...,4]>0.5][...,0:5]
            return pred_final.view(nB, nA*nG*nG, nCH), None
        else:
            # training, convert final predictions to be normalized
            pred_final[..., 2:4] /= img_size
            # force the normalized w and h to between 0 and 1
            pred_final[..., 2:4].clamp_(max=1)

        # tic = time()
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        # assert (labels >= 0).all() and (labels[...,1:5] <= 1).all()

        tcls_all = labels[:,:,0].long()
        tx_all, ty_all = labels[:,:,1] * nG, labels[:,:,2] * nG # 0-nG
        tw_all, th_all = labels[:,:,3], labels[:,:,4] # normalized 0-1

        ti_all = tx_all.long()
        tj_all = ty_all.long()

        norm_anch_wh = anchors[:,0:2].cpu() / img_size # normalized
        norm_anch_00wh = self.anch_00wh_all.clone()
        norm_anch_00wh[:,2:4] /= img_size # normalized
        # torch.cuda.synchronize()
        # self.time_dic['process_label_anchor'] += time() - tic

        # traverse all images in a batch
        valid_gt_num = 0
        obj_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.bool)
        penalty_mask = torch.ones(nB, nA, nG, nG, dtype=torch.bool)
        tgt_scale = torch.zeros(nB, nA, nG, nG, 2)
        target = torch.zeros(nB, nA, nG, nG, nCH)
        for b in range(nB):
            n = int(nlabel[b]) # number of ground truths in b'th image
            if n == 0:
                # no ground truth
                continue
            gt_boxes = torch.zeros(n, 4)
            gt_boxes[:, 2] = tw_all[b, :n] # normalized 0-1
            gt_boxes[:, 3] = th_all[b, :n] # normalized 0-1

            # tic = time()
            # calculate iou between truth and reference anchors
            anchor_ious = bboxes_iou(gt_boxes, norm_anch_00wh, xyxy=False)
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
            # torch.cuda.synchronize()
            # self.time_dic['gt_anchor_select'] += time() - tic
            
            best_n = best_n[valid_mask]
            truth_i = ti_all[b, :n][valid_mask]
            truth_j = tj_all[b, :n][valid_mask]

            gt_boxes[:, 0] = tx_all[b, :n] / nG # normalized 0-1
            gt_boxes[:, 1] = ty_all[b, :n] / nG # normalized 0-1

            # tic = time()
            # gt_boxes e.g. shape(11,4)
            pred_ious = bboxes_iou(pred_final[b,...,0:4].view(-1,4), gt_boxes, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            ignore_mask = (pred_best_iou > self.ignore_thre)
            ignore_mask = ignore_mask.view(pred_final[b,...,0:4].shape[:3])
            # set mask to zero (ignore) if pred matches a truth more than 0.7
            penalty_mask[b] = ~ignore_mask
            # penalty_mask = 1 -> give penalty
            # torch.cuda.synchronize()
            # self.time_dic['pred_gt_ignore'] += time() - tic

            # tic = time()
            penalty_mask[b,best_n,truth_j,truth_i] = 1
            obj_mask[b,best_n,truth_j,truth_i] = 1
            target[b,best_n,truth_j,truth_i,0] = tx_all[b,:n][valid_mask] - tx_all[b,:n][valid_mask].floor()
            target[b,best_n,truth_j,truth_i,1] = ty_all[b,:n][valid_mask] - ty_all[b,:n][valid_mask].floor()
            target[b,best_n,truth_j,truth_i,2] = torch.log(tw_all[b,:n][valid_mask]/norm_anch_wh[best_n,0] + 1e-16)
            target[b,best_n,truth_j,truth_i,3] = torch.log(th_all[b,:n][valid_mask]/norm_anch_wh[best_n,1] + 1e-16)
            target[b,best_n,truth_j,truth_i,4] = 1 # objectness confidence
            if self.n_cls > 0:
                target[b,best_n,truth_j,truth_i,5+tcls_all[b,:n][valid_mask]] = 1
            # smaller objects have higher losses
            tgt_scale[b,best_n,truth_j,truth_i,:] = torch.sqrt(2 - tw_all[b,:n][valid_mask]*th_all[b,:n][valid_mask]).unsqueeze(1)
            # torch.cuda.synchronize()
            # self.time_dic['set_target'] += time() - tic

        # tic = time()
        # move the tagerts to GPU
        obj_mask = obj_mask.to(device=device)
        penalty_mask = penalty_mask.to(device=device)
        tgt_scale = tgt_scale.to(device=device)
        target = target.to(device=device)

        xywh_loss_weight = tgt_scale[obj_mask]
        # weighted BCEloss for x,y,w,h
        loss_xy = tnf.binary_cross_entropy(xy_offset[obj_mask], target[...,0:2][obj_mask],
                            weight=xywh_loss_weight*xywh_loss_weight, reduction='sum')
        wh_pred = wh_scale[obj_mask] * xywh_loss_weight
        wh_target = target[...,2:4][obj_mask] * xywh_loss_weight
        loss_wh = tnf.mse_loss(wh_pred, wh_target, reduction='sum')
        loss_obj = self.loss4obj(conf[penalty_mask], target[...,4][penalty_mask])
        if self.n_cls > 0:
            loss_cls = tnf.binary_cross_entropy(classes[obj_mask], target[...,5:][obj_mask],
                                                reduction='sum')
        else:
            loss_cls = 0

        loss = loss_xy + 0.5*loss_wh + loss_obj + loss_cls
        # torch.cuda.synchronize()
        # self.time_dic['calculate_loss'] += time() - tic

        # logging
        ngt = valid_gt_num + 1e-16
        self.loss_str = f'yolo_{nG} total {int(ngt)} objects: ' \
                        f'xy/gt {loss_xy/ngt:.3f}, wh/gt {loss_wh/ngt:.3f}' \
                        f', conf {loss_obj:.3f}, class {loss_cls:.3f}'
        self.gt_num = valid_gt_num
        return None, loss
