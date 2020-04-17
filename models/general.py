import json
import torch

from .registry import get_backbone, get_fpn, get_rpn, get_det_layer
from utils.structures import ImageObjects


def name_to_model(model_name):
    cfg = json.load(open(f'./configs/{model_name}.json', 'r'))

    if cfg['base'] == 'OneStageBBox':
        model = OneStageBBox(cfg)
    else:
        raise Exception('Unknown model name')
    
    return model, cfg


class OneStageBBox(torch.nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        self.backbone = get_backbone(cfg)
        self.fpn = get_fpn(cfg)
        self.rpn = get_rpn(cfg)
        
        det_layer = get_det_layer(cfg)
        self.det_layers = torch.nn.ModuleList()
        for level_i in range(len(cfg['model.fpn.out_channels'])):
            self.det_layers.append(det_layer(level_i=level_i, cfg=cfg))

        self.check_gt_assignment = cfg.get('general.check_gt_assignment', False)
        self.bb_format = cfg.get('general.pred_bbox_format', 'cxcywh')
        self.input_format = cfg['general.input_format']

    def forward(self, x, labels:list=None):
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
                total_gt_num = sum([t.shape[0] for t in labels])
                assigned = sum(branch._assigned_num for branch in self.det_layers)
                assert assigned == total_gt_num
            self.loss_str = ''
            for m in self.det_layers:
                self.loss_str += m.loss_str + '\n'
            loss = sum(losses_all)
            return loss
