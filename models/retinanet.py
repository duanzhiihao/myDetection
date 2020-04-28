import numpy as np
import torch
import torch.nn.functional as tnf
import fvcore.nn

from utils.bbox_ops import bboxes_iou
from utils.structures import ImageObjects


bce_w_logits = tnf.binary_cross_entropy_with_logits

class RetinaLayer(torch.nn.Module):
    '''
    calculate the output boxes and losses
    '''
    def __init__(self, level_i: int, cfg: dict):
        super().__init__()
        # stride at this level
        stride = cfg['model.fpn.out_strides'][level_i]
        # generate anchors based on scales and ratios
        base_size = cfg['model.retina.anchor.base'] * stride
        scales = cfg['model.retina.anchor.scales']
        ratios = cfg['model.retina.anchor.ratios']
        anchors = []
        for sc in scales:
            for rt in ratios:
                anchors.append((base_size*sc*rt[0], base_size*sc*rt[1]))
        self.anchor_wh = torch.Tensor(anchors)
        self.num_anchors = len(anchors)
        self.positive_thres = cfg['model.retina.anchor.positive_threshold']
        self.negative_thres = cfg['model.retina.anchor.negative_threshold']
        self.stride = stride
        self.n_cls = cfg['general.num_class']
        self.pred_bbox_format = cfg['general.pred_bbox_format']
        if self.pred_bbox_format == 'cxcywhd':
            from .losses import get_angle_loss
            self.loss_angle = get_angle_loss(cfg['model.angle.loss_name'],
                                             reduction='sum')
        self.n_bbparam = cfg['general.bbox_param']
        self.last_img_size = None

    def forward(self, raw: dict, img_size, labels=None):
        stride = self.stride
        img_h, img_w = img_size
        nA = self.num_anchors # number of anchors
        nH, nW = int(img_h / stride), int(img_w / stride)
        nCls = self.n_cls
        
        t_xywh = raw['bbox']
        cls_logits = raw['class']
        nB = t_xywh.shape[0] # batch size
        assert t_xywh.shape == (nB, nA, nH, nW, self.n_bbparam)
        assert cls_logits.shape == (nB, nA, nH, nW, nCls)
        device = t_xywh.device

        def _compute_anchors(dvc):
            # generate anchor boxes on a specific device
            a_cx = torch.arange(stride/2, img_w, stride, device=dvc).view(1,1,1,nW)
            a_cy = torch.arange(stride/2, img_h, stride, device=dvc).view(1,1,nH,1)
            a_wh = self.anchor_wh.view(1, nA, 1, 1, 2).to(device=dvc)
            return a_cx, a_cy, a_wh

        if labels is None:
            # -------------------- logits to prediction --------------------
            # cx, cy, w, h
            a_cx, a_cy, a_wh = _compute_anchors(dvc=device)
            p_xywh = torch.empty_like(t_xywh).contiguous()
            p_xywh[..., 0] = a_cx + t_xywh[..., 0] * a_wh[..., 0]
            p_xywh[..., 1] = a_cy + t_xywh[..., 1] * a_wh[..., 1]
            p_xywh[..., 2:4] = torch.exp(t_xywh[..., 2:4]) * a_wh
            p_xywh[..., 0:4].clamp_(min=1, max=max(img_size))
            if self.pred_bbox_format == 'cxcywhd':
                p_xywh[..., 4] = torch.sigmoid(t_xywh[..., 4])*360 - 180
            # classes
            p_cls = torch.sigmoid(cls_logits)
            cls_score, cls_idx = torch.max(p_cls, dim=-1)
            preds = {
                'bbox': p_xywh.view(nB, nA*nH*nW, self.n_bbparam).cpu(),
                'class_idx': cls_idx.view(nB, nA*nH*nW).cpu(),
                'score': cls_score.view(nB, nA*nH*nW).cpu(),
            }
            return preds, None
        
        a_cx, a_cy, a_wh = _compute_anchors(dvc=torch.device('cpu'))
        # a_cx, a_cy, a_wh = [a.squeeze(0) for a in [a_cx, a_cy, a_wh]]
        anch_bbs = torch.cat([
            a_cx.view(1, 1, nW, 1).expand(nA, nH, nW, 1),
            a_cy.view(1, nH, 1, 1).expand(nA, nH, nW, 1),
            a_wh.view(nA, 1, 1, 2).expand(nA, nH, nW, 2)], dim=-1)
        loss_xywh = 0
        loss_cls = 0
        total_pos_num = 0
        total_sample_num = 0
        for b in range(nB):
            im_labels = labels[b]
            assert isinstance(im_labels, ImageObjects)
            im_labels.sanity_check()
            assert self.pred_bbox_format == im_labels._bb_format

            if len(im_labels) == 0:
                tgt_cls = torch.zeros(nA, nH, nW, nCls, device=device)
                im_loss_cls = bce_w_logits(cls_logits[b], tgt_cls, reduction='sum')
                loss_cls = loss_cls + im_loss_cls
                continue
            
            gt_bbs = im_labels.bboxes
            ious = bboxes_iou(anch_bbs.view(-1, 4), gt_bbs[:,:4], xyxy=False)
            iou_with_gt, gt_idx = ious.max(dim=1)
            iou_with_gt = iou_with_gt.view(nA, nH, nW)
            gt_idx = gt_idx.view(nA, nH, nW)
            M_pos = (iou_with_gt > self.positive_thres) # positive sample mask
            M_neg = (iou_with_gt < self.negative_thres) # negative sample mask
            num_pos_sample = M_pos.sum()
            total_pos_num += num_pos_sample
            total_sample_num += nA*nH*nW

            # set bbox target
            tgt_xywh = torch.zeros(nA, nH, nW, 4)
            gt_bbs = gt_bbs[gt_idx, :]
            tgt_xywh[...,0:2] = (gt_bbs[...,0:2]-anch_bbs[...,0:2]) / anch_bbs[...,2:4]
            tgt_xywh[...,2:4] = torch.log(gt_bbs[...,2:4] / anch_bbs[...,2:4] + 1e-8)
            # set class target
            tgt_cls = torch.zeros(nA, nH, nW, nCls)
            tgt_cls[M_pos, im_labels.cats[gt_idx[M_pos]]] = 1
            # find the predictions which are not good enough
            high_enough = np.log(0.95 / (1 - 0.95))
            need_higher = M_pos & (cls_logits[b] < high_enough)
            low_enough = np.log(0.01 / (1 - 0.01))
            need_lower = M_neg & (cls_logits[b] > low_enough)
            # ignore the predictions which are already good enough
            cls_penalty_mask = need_higher | need_lower
            # Set angle target.
            if self.pred_bbox_format == 'cxcywhd':
                tgt_angle = torch.zeros(nA, nH, nW)
                # Use radian when calculating the angle loss
                tgt_angle = gt_bbs[..., 4] / 180 * np.pi
                tgt_angle = tgt_angle.to(device)
            tgt_xywh = tgt_xywh.to(device)
            tgt_cls = tgt_cls.to(device)
            # bbox loss
            if num_pos_sample > 0:
                # import matplotlib.pyplot as plt
                # for ia in range(nA):
                #     print(a_wh.squeeze()[ia,:])
                #     mask = M_pos[ia, :, :].numpy()
                #     plt.imshow(mask, cmap='gray')
                #     plt.show()
                im_loss_xywh = fvcore.nn.smooth_l1_loss(t_xywh[b][M_pos][:,0:4],
                                tgt_xywh[M_pos, :], beta=0.1, reduction='sum')
                if self.pred_bbox_format == 'cxcywhd':
                    p_angle = torch.sigmoid(t_xywh[b][M_pos][:,4])*2*np.pi - np.pi
                    im_loss_angle = self.loss_angle(p_angle, tgt_angle[M_pos])
                    im_loss_xywh = im_loss_xywh + im_loss_angle
                # im_loss_xywh = tnf.mse_loss(t_xywh[b, M_pos, :],
                #                 tgt_xywh[M_pos, :], reduction='sum')
                loss_xywh = loss_xywh + im_loss_xywh
            # class loss
            # im_loss_cls = fvcore.nn.sigmoid_focal_loss(cls_logits[b, cls_penalty_mask],
            #         tgt_cls[cls_penalty_mask], alpha=0.25, gamma=2, reduction='sum')
            im_loss_cls = bce_w_logits(cls_logits[b, cls_penalty_mask],
                            tgt_cls[cls_penalty_mask], reduction='sum')
            loss_cls = loss_cls + im_loss_cls # / (num_pos_sample + 1)
        loss = (loss_xywh + loss_cls) / nB

        # logging
        self.loss_str = f'level_{nH}x{nW} pos {total_pos_num}/{total_sample_num}: ' \
                        f'xywh {loss_xywh:.3f}, class {loss_cls:.3f}'
        return None, loss


    # def compute_anchor_boxes(self, img_size):
    #     stride = self.stride
    #     img_h, img_w = img_size
    #     nA = self.num_anchors # number of anchors
    #     nH, nW = int(img_h / stride), int(img_w / stride)
    #     if img_size != self.last_img_size:
    #         # If img_size is different from the previous, compute new anchor boxes
    #         y_ = torch.arange(stride/2, img_h, stride)
    #         x_ = torch.arange(stride/2, img_w, stride)
    #         # mesh_y, mesh_x = torch.meshgrid(y_, x_)
    #         anch_cx = x_.view(1, 1, 1, nW, 1).expand(1, nA, nH, nW, 1)
    #         anch_cy = y_.view(1, 1, nH, 1, 1).expand(1, nA, nH, nW, 1)
    #         anch_wh = self.anchor_wh.view(1, nA, 1, 1, 2).expand(1, nA, nH, nW, 2)
    #         anch_bbs = [anch_cx, anch_cy, anch_wh]
    #         self.last_anchor_boxes = anch_bbs
    #         self.last_img_size = img_size
    #     # Otherwise, directly return the previously calculated anchor boxes
    #     return self.last_anchor_boxes
        