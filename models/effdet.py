import torch
import torch.nn as nn

from .backbones import EfNetBackbone
from .fpns import get_bifpn
from .rpns import EfDetHead


class EfficientDet(nn.Module):
    '''
    Args:
        model_id: str, 'd0', 'd1', ..., 'd7'
    '''
    model_configs = {
        'd0': {'RES': 512, 'FPN_CH': 64, 'FPN_NUM': 3, 'HEAD_NUM': 3},
    }
    def __init__(self, model_id='d0', num_class=80):
        super().__init__()
        assert model_id in EfficientDet.model_configs, 'Unsupported model'
        _cfg = EfficientDet.model_configs[model_id]

        backbone_name = f'efficientnet-b' + model_id[1]
        self.backbone = EfNetBackbone(backbone_name, _cfg['FPN_CH'])

        self.fpn = get_bifpn(self.backbone.feature_chs, out_ch=_cfg['FPN_CH'],
                             repeat_num=_cfg['FPN_NUM'], fusion_method='linear')
        
        fpn_chs = [_cfg['FPN_CH'] for _ in self.backbone.feature_chs]
        nC = num_class
        nA = 9
        self.rpn = EfDetHead(fpn_chs, _cfg['HEAD_NUM'], cls_ch=nC*nA, bbox_ch=nA*4)

    def forward(self, img_batch, labels=None):
        features = self.backbone(img_batch)
        features = self.fpn(features)
        raw_preds = self.rpn(features)

        debug = 1
        return 0
