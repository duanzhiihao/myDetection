from typing import List
import torch

import models.registry as registry
from utils.structures import ImageObjects


class SimpleVOD(torch.nn.Module):
    '''
    A simple video object detection class
    '''
    def __init__(self, cfg: dict):
        super().__init__()

        self.backbone = registry.get_backbone(cfg)
        self.fpn      = registry.get_fpn(cfg)
        self.agg      = registry.get_agg(cfg)
        self.rpn      = registry.get_rpn(cfg)
        
        det_layer = registry.get_det_layer(cfg)
        self.det_layers = torch.nn.ModuleList()
        for level_i in range(len(cfg['model.fpn.out_channels'])):
            self.det_layers.append(det_layer(level_i=level_i, cfg=cfg))

        self.check_gt_assignment = cfg.get('train.check_gt_assignment', False)
        self.bb_format = cfg.get('general.pred_bbox_format')
        self.input_format = cfg['general.input_format']

        self.hidden = None
    
    def clear_hidden_state(self):
        self.hidden = None

    def forward(self, x, is_start: torch.BoolTensor=None,
                labels: List[ImageObjects]=None):
        '''
        Forwar pass

        Args:
            x: a batch of images, e.g. shape(8,3,608,608)
            is_start: a batch of bool tensors indicating if the input x is the \
                start of a video.
            labels: a batch of ground truth
        '''
        assert x.dim() == 4
        if is_start is None:
            is_start = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        assert is_start.dim() == 1 and is_start.shape[0] == x.shape[0]
        self.img_size = x.shape[2:4]

        # go through the backbone and the feature payamid network
        features = self.backbone(x)
        features = self.fpn(features)
        # feature aggregation
        features = self.agg(features, self.hidden, is_start)
        self.hidden = [f.detach() for f in features]
        # prediction
        all_branch_preds = self.rpn(features)
        
        dts_all = []
        losses_all = []
        for i, raw_preds in enumerate(all_branch_preds):
            dts, loss = self.det_layers[i](raw_preds, self.img_size, labels)
            dts_all.append(dts)
            losses_all.append(loss)

        if labels is None:
            batch_bbs = torch.cat([d['bbox'] for d in dts_all], dim=1)
            batch_cls_idx = torch.cat([d['class_idx'] for d in dts_all], dim=1)
            batch_scores = torch.cat([d['score'] for d in dts_all], dim=1)

            batch_pred_objects = []
            # iterate over every image in the batch
            for bbs, cls_idx, scores in zip(batch_bbs, batch_cls_idx, batch_scores):
                # initialize the pred objects in current image
                p_objs = ImageObjects(bboxes=bbs, cats=cls_idx, scores=scores,
                                      bb_format=self.bb_format)
                batch_pred_objects.append(p_objs)
            return batch_pred_objects
        else:
            if self.check_gt_assignment:
                total_gt_num = sum([len(t) for t in labels])
                assigned = sum(branch._assigned_num for branch in self.det_layers)
                assert assigned == total_gt_num, f'{assigned} != {total_gt_num}'
            self.loss_str = ''
            for m in self.det_layers:
                self.loss_str += m.loss_str + '\n'
            loss = sum(losses_all)
            return loss
