import torch
import torch.nn as nn
import torch.nn.functional as tnf

from .backbones import EfNetBackbone
from .fpns import get_bifpn
from .rpns import EfDetHead
from utils.iou_funcs import bboxes_iou
from utils.structures import ImageObjects


class EfficientDet(nn.Module):
    '''
    Args:
        model_id: str, 'd0', 'd1', ..., 'd7'
    '''
    model_configs = {
        'd0': {'RES': 512, 'FPN_CH': 64, 'FPN_NUM': 3, 'HEAD_NUM': 3},
    }
    def __init__(self, cfg: dict):
        super().__init__()
        # Get config of the model
        cfg.update(EfficientDet.model_configs[cfg['model_id']])

        # Initialize backbone
        backbone_name = f'efficientnet-b' + cfg['model_id'][1]
        self.backbone = EfNetBackbone(backbone_name, cfg['FPN_CH'], C6C7=cfg['C6C7'])
        # Initialize FPN
        self.fpn = get_bifpn(self.backbone.feature_chs, out_ch=cfg['FPN_CH'],
                             repeat_num=cfg['FPN_NUM'], fusion_method='linear')
        # Initialize bbox and class network
        fpn_chs = [cfg['FPN_CH'] for _ in self.backbone.feature_chs]
        nC = cfg['num_class']
        nA = cfg['num_anchor_per_level']
        self.rpn = EfDetHead(fpn_chs, cfg['HEAD_NUM'], cls_ch=nC*nA, bbox_ch=nA*4)
        # Initialize prediction layers
        self.bb_layers = nn.ModuleList()
        if cfg['pred_layer'] == 'YOLO':
            pred_layer = YOLOLayer
        else:
            raise NotImplementedError()
        strides = [8, 16, 32, 64, 128] if cfg['C6C7'] else [8, 16, 32]
        for level_i, s in enumerate(strides):
            self.bb_layers.append(pred_layer(strides_all=strides, stride=s,
                                             level=level_i, **cfg))
        
        self.input_format = cfg['input_format']

    def forward(self, img_batch, labels=None):
        assert img_batch.dim() == 4
        self.img_size = img_batch.shape[2:4]

        features = self.backbone(img_batch)
        features = self.fpn(features)
        all_level_preds = self.rpn(features)

        dts_all = []
        losses_all = []
        for i, raw_preds in enumerate(all_level_preds):
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
        indices = [[6,7,8], [3,4,5], [0,1,2]]
        self.anchors_all = torch.Tensor(anchors).float()

        level_i = kwargs['level']
        self.stride = kwargs['stride']
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
        # self.loss4conf = FocalBCE(reduction='sum')
        # self.loss4conf = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, raw: dict, img_size, labels=None):
        assert isinstance(raw, dict)
        t_xywh = raw.pop('bbox')
        cls_logits = raw.pop('class')
        assert not raw
        device = t_xywh.device
        nB = t_xywh.shape[0] # batch size
        nA = self.num_anchors # number of anchors
        nH, nW = t_xywh.shape[2:4] # prediction grid size

        t_xywh = t_xywh.view(nB,nA,4,nH,nW).permute(0, 1, 3, 4, 2)
        cls_logits = cls_logits.view(nB,nA,self.n_cls,nH,nW).permute(0, 1, 3, 4, 2)
        
        # ----------------------- logits to prediction -----------------------
        y_ = torch.arange(nH, dtype=torch.float, device=device)
        x_ = torch.arange(nW, dtype=torch.float, device=device)
        mesh_y, mesh_x = torch.meshgrid(y_, x_)
        # calculate xywh from transformed version
        p_xywh = t_xywh.clone().detach()
        p_xywh[...,0] = (torch.sigmoid(p_xywh[...,0]) + mesh_x) * self.stride
        p_xywh[...,1] = (torch.sigmoid(p_xywh[...,1]) + mesh_y) * self.stride
        anch_wh = self.anchors.view(1,nA,1,1,2).to(device=device)
        p_xywh[...,2:4] = torch.exp(p_xywh[...,2:4]) * anch_wh
        p_xywh = p_xywh.view(nB, -1, 4).cpu()

        if labels is None:
            # ---------------------------- testing ----------------------------
            # Logistic activation for categories
            p_cls = torch.sigmoid(cls_logits).cpu()
            cls_score, cls_idx = torch.max(p_cls, dim=-1)
            preds = {
                'bbox': p_xywh,
                'class_idx': cls_idx.view(nB, -1),
                'conf': cls_score.view(nB, -1),
            }
            return preds, None

        # ------------------------------ training ------------------------------
        assert isinstance(labels, list)
        # traverse all images in a batch
        valid_gt_num = 0
        gt_mask = torch.zeros(nB, nA, nH, nW, dtype=torch.bool)
        tgt_xywh = torch.zeros(nB, nA, nH, nW, 4)
        tgt_cls = torch.zeros(nB, nA, nH, nW, self.n_cls)
        cls_loss_mask = torch.ones(nB, nA, nH, nW, dtype=torch.bool)
        weighted = torch.zeros(nB, nA, nH, nW)
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
            cls_loss_mask[b] = (iou_with_gt < self.ignore_thre).view(nA,nH,nW)
            # cls_loss_mask = 1 -> give penalty

            im_labels = im_labels[valid_mask,:]
            grid_tx = im_labels[:,1] / self.stride
            grid_ty = im_labels[:,2] / self.stride
            ti, tj = grid_tx.long(), grid_ty.long()
            tn = best_n[valid_mask] # target anchor box number
            
            gt_mask[b,tn,tj,ti] = 1
            cls_loss_mask[b,tn,tj,ti] = 1
            tgt_xywh[b,tn,tj,ti,0] = grid_tx - grid_tx.floor()
            tgt_xywh[b,tn,tj,ti,1] = grid_ty - grid_ty.floor()
            tgt_xywh[b,tn,tj,ti,2] = torch.log(im_labels[:,3]/self.anchors[tn,0] + 1e-8)
            tgt_xywh[b,tn,tj,ti,3] = torch.log(im_labels[:,4]/self.anchors[tn,1] + 1e-8)
            tgt_cls[b, tn, tj, ti, im_labels[:,0].long()] = 1
            # smaller objects have higher losses
            img_area = img_size[0] * img_size[1]
            weighted[b,tn,tj,ti] = 2 - im_labels[:,3]*im_labels[:,4] / img_area

        # move the tagerts to GPU
        gt_mask = gt_mask.to(device=device)
        cls_loss_mask = cls_loss_mask.to(device=device)
        weighted = weighted.unsqueeze(-1).to(device=device)
        tgt_xywh = tgt_xywh.to(device=device)
        tgt_cls = tgt_cls.to(device=device)

        bce_logits = tnf.binary_cross_entropy_with_logits
        # weighted BCE loss for x,y
        loss_xy = bce_logits(t_xywh[...,0:2][gt_mask], tgt_xywh[...,0:2][gt_mask],
                             weight=weighted[gt_mask], reduction='sum')
        # weighted squared error for w,h
        loss_wh = (t_xywh[...,2:4][gt_mask] - tgt_xywh[...,2:4][gt_mask]).pow(2)
        loss_wh = (weighted[gt_mask] * loss_wh).sum()
        loss_cls = bce_logits(cls_logits[cls_loss_mask], tgt_cls[cls_loss_mask],
                              reduction='sum')
        loss = loss_xy + 0.5*loss_wh + loss_cls

        # logging
        ngt = valid_gt_num + 1e-16
        self.loss_str = f'level_{nH}x{nW} total {int(ngt)} objects: ' \
                        f'xy/gt {loss_xy/ngt:.3f}, wh/gt {loss_wh/ngt:.3f}' \
                        f', class {loss_cls:.3f}'
        self.gt_num = valid_gt_num
        return None, loss
