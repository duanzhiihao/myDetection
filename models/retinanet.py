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
        assert t_xywh.shape == (nB, nA, nH, nW, 4)
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
            p_xywh = torch.empty_like(t_xywh)
            p_xywh[..., 0] = a_cx + t_xywh[..., 0] * a_wh[..., 0]
            p_xywh[..., 1] = a_cy + t_xywh[..., 1] * a_wh[..., 1]
            p_xywh[..., 2:4] = torch.exp(t_xywh[..., 2:4]) * a_wh
            p_xywh.clamp_(min=1, max=max(img_size))
            # classes
            p_cls = torch.sigmoid(cls_logits)
            cls_score, cls_idx = torch.max(p_cls, dim=-1)
            preds = {
                'bbox': p_xywh.view(nB, nA*nH*nW, 4).cpu(),
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

            if len(im_labels) == 0:
                tgt_cls = torch.zeros(nA, nH, nW, nCls, device=device)
                im_loss_cls = bce_w_logits(cls_logits[b], tgt_cls, reduction='sum')
                loss_cls = loss_cls + im_loss_cls / (num_pos_sample + 1)
                continue
            
            gt_bbs = im_labels.bboxes
            ious = bboxes_iou(anch_bbs.view(-1, 4), gt_bbs, xyxy=False)
            iou_with_gt, gt_idx = ious.max(dim=1)
            iou_with_gt = iou_with_gt.view(nA, nH, nW)
            gt_idx = gt_idx.view(nA, nH, nW)
            pos_idx = (iou_with_gt > self.positive_thres)
            neg_idx = (iou_with_gt < self.negative_thres)
            num_pos_sample = pos_idx.sum()
            total_pos_num += num_pos_sample
            total_sample_num += nA*nH*nW

            # set bbox target
            tgt_xywh = torch.zeros(nA, nH, nW, 4)
            gt_bbs = gt_bbs[gt_idx, :]
            tgt_xywh[...,0:2] = (gt_bbs[...,0:2]-anch_bbs[...,0:2]) / anch_bbs[...,2:4]
            tgt_xywh[...,2:4] = torch.log(gt_bbs[...,2:4] / anch_bbs[...,2:4] + 1e-8)
            # set class target
            tgt_cls = torch.zeros(nA, nH, nW, nCls)
            tgt_cls[pos_idx, im_labels.cats[gt_idx[pos_idx]]] = 1
            penalty_mask = (pos_idx | neg_idx)

            tgt_xywh = tgt_xywh.to(device)
            tgt_cls = tgt_cls.to(device)
            # bbox loss
            if num_pos_sample > 0:
                # import numpy as np
                # import matplotlib.pyplot as plt
                # for i, mask in enumerate(pos_idx):
                #     print(self.anchor_wh[i])
                #     plt.figure(); plt.imshow(mask.numpy(), cmap='gray')
                #     plt.show()
                # bg = np.zeros((img_h,img_w,3))
                # debug = ImageObjects(anch_bbs[pos_idx], 
                #                      cats=torch.zeros(30).long())
                # debug_tgt_xywh = tgt_xywh.cpu().unsqueeze(0)
                # p_xywh = torch.empty_like(debug_tgt_xywh)
                # p_xywh[..., 0] = a_cx + debug_tgt_xywh[..., 0] * a_wh[..., 0]
                # p_xywh[..., 1] = a_cy + debug_tgt_xywh[..., 1] * a_wh[..., 1]
                # p_xywh[..., 2:4] = torch.exp(debug_tgt_xywh[..., 2:4]) * a_wh
                # p_xywh.clamp_(min=1, max=max(img_size))
                # debug = ImageObjects(p_xywh.squeeze(0)[pos_idx].view(-1,4),
                #                 cats=torch.zeros(pos_idx.sum()).long())
                # debug.draw_on_np(bg)
                # plt.figure(); plt.imshow(bg); plt.show()
                # im_loss_xywh = fvcore.nn.smooth_l1_loss(t_xywh[b, pos_idx, :],
                #                 tgt_xywh[pos_idx, :], beta=0.1, reduction='mean')
                im_loss_xywh = tnf.mse_loss(t_xywh[b, pos_idx, :],
                                tgt_xywh[pos_idx, :], reduction='sum')
                loss_xywh = loss_xywh + im_loss_xywh
            # class loss
            # im_loss_cls = fvcore.nn.sigmoid_focal_loss(cls_logits[b, penalty_mask],
            #             tgt_cls[penalty_mask], alpha=0.25, gamma=2, reduction='sum')
            im_loss_cls = bce_w_logits(cls_logits[b, penalty_mask],
                            tgt_cls[penalty_mask], reduction='sum')
            loss_cls = loss_cls + im_loss_cls / (num_pos_sample + 1)
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
        