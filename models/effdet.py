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
        'd1': {'RES': 640, 'FPN_CH': 88, 'FPN_NUM': 4, 'HEAD_NUM': 3},
        'd2': {'RES': 768, 'FPN_CH': 112, 'FPN_NUM': 5, 'HEAD_NUM': 3},
        'd3': {'RES': 896, 'FPN_CH': 160, 'FPN_NUM': 6, 'HEAD_NUM': 4},
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
        if cfg['pred_layer'] == 'EffDet':
            pred_layer = EffLayer
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


class EffLayer(nn.Module):
    '''
    '''
    def __init__(self, **kwargs):
        level_i = kwargs['level']
        self.stride = kwargs['strides_all'][level_i]
        self.n_cls = kwargs['num_class']

    def forward(self, raw: dict, img_size, labels=None):
        pass