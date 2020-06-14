import numpy as np
import torch
import torch.nn as nn

from .modules import SeparableConv2d, Swish


class YOLOHead(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.n_anch   = cfg['model.yolo.num_anchor_per_level']
        self.n_cls    = cfg['general.num_class']
        self.bb_param = cfg.get('general.bbox_param', 4)
        self.heads    = nn.ModuleList()
        out_ch = (self.bb_param + 1 + self.n_cls) * self.n_anch
        for ch in cfg['model.fpn.out_channels']:
            self.heads.append(nn.Conv2d(ch, out_ch, 1, stride=1, padding=0))

    def forward(self, features):
        nBp = self.bb_param
        all_level_preds = []
        for module, P in zip(self.heads, features):
            preds = module(P)
            nB, _, nH, nW = preds.shape
            preds = preds.view(nB, self.n_anch, nBp+1+self.n_cls, nH, nW)
            
            raw = {
                'bbox':  preds[:, :, 0:nBp, :, :].permute(0, 1, 3, 4, 2),
                'conf':  preds[:, :, nBp:nBp+1, :, :].permute(0, 1, 3, 4, 2),
                'class': preds[:, :, nBp+1:, :, :].permute(0, 1, 3, 4, 2),
            }
            all_level_preds.append(raw)
        
        return all_level_preds


# Source: https://github.com/tianzhi0549/FCOS
class FCOSHead(nn.Module):
    def __init__(self, in_ch, num_class):
        """
        Arguments:
            in_ch (int): number of channels of the input feature
        """
        super().__init__()
        num_convs = 4

        cls_tower = []
        bbox_tower = []
        for i in range(num_convs):
            cls_tower.append(nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1))
            cls_tower.append(nn.BatchNorm2d(in_ch))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1))
            bbox_tower.append(nn.BatchNorm2d(in_ch))
            bbox_tower.append(nn.ReLU())
        self.cls_tower  = nn.Sequential(*cls_tower)
        self.cls_pred   = nn.Conv2d(in_ch, num_class, 3, stride=1, padding=1)
        self.bbox_tower = nn.Sequential(*bbox_tower)
        self.bbox_pred  = nn.Conv2d(in_ch, 4, 3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_ch, 1, 3, stride=1, padding=1)

        # initialization
        # for modules in [self.cls_tower, self.bbox_tower,
        #                 self.cls_pred, self.bbox_pred, self.centerness]:
        #     for l in modules.modules():
        #         if isinstance(l, nn.Conv2d):
        #             nn.init.normal_(l.weight, std=0.01)
        #             nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        # prior_prob = torch.Tensor([0.01])[0]
        # bias_value = -torch.log((1 - prior_prob) / prior_prob)
        # nn.init.constant_(self.cls_pred.bias, bias_value)

        # self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        # cls_logits = []
        # bbox_reg = []
        # centerness = []
        all_branch_preds = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            conf_pred = self.centerness(box_tower)
            # bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            bbox_pred = self.bbox_pred(box_tower)
            cls_pred  = self.cls_pred(cls_tower)
            # bbox_reg.append(bbox_pred)

            raw = {
                'bbox': bbox_pred,
                'class': cls_pred,
                'conf': conf_pred,
            }
            all_branch_preds.append(raw)

        return all_branch_preds

# class Scale(nn.Module):
#     def __init__(self, init_value=1.0):
#         super(Scale, self).__init__()
#         self.scale = nn.Parameter(torch.FloatTensor([init_value]))

#     def forward(self, x):
#         return x * self.scale


class EfDetHead(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        n_cls          = cfg['general.num_class']
        n_anch         = cfg['model.effrpn.num_anchor_per_level']
        feature_chs    = cfg['model.fpn.out_channels']
        repeat         = cfg['model.effrpn.repeat_num']
        bb_param       = cfg.get('general.bbox_param', 4)
        enable_conf    = cfg['model.effrpn.enable_conf']
        bbox_last_type = cfg.get('model.effrpn.bbox_last', 'default')
        cls_last_type  = cfg.get('model.effrpn.cls_last', 'spconv')

        self.class_nets = nn.ModuleList()
        self.bbox_nets = nn.ModuleList()
        cls_ch = n_anch * (1 + n_cls) if enable_conf else n_anch * n_cls
        for ch in feature_chs:
            bb_net = [spconv3x3_bn_swish(ch) for _ in range(repeat)]
            if bbox_last_type == 'default':
                bbox_last = SeparableConv2d(ch, n_anch*bb_param, 3, 1, padding=1)
            elif bbox_last_type == 'lr_tb':
                assert n_anch == 1 and bb_param == 4
                bbox_last = _LR_TB_last(ch)
            else: raise NotImplementedError()
            bb_net.append(bbox_last)
            bb_net = nn.Sequential(*bb_net)
            self.bbox_nets.append(bb_net)

            cls_net = [spconv3x3_bn_swish(ch) for _ in range(repeat)]
            # Initialize the final layer bias by -np.log((1 - 0.01) / 0.01)
            # so that the initial confidence are close to 0.01
            if cls_last_type == 'spconv':
                cls_last = SeparableConv2d(ch, cls_ch, 3, 1, padding=1)
                cls_last.pointwise.weight.data.normal_(mean=0, std=0.1)
                cls_last.pointwise.bias.data.fill_(-np.log((1 - 0.01) / 0.01))
            elif cls_last_type == 'conv':
                cls_last = nn.Conv2d(ch, cls_ch, 3, 1, padding=1)
                cls_last.weight.data.normal_(mean=0, std=0.1)
                cls_last.bias.data.fill_(-np.log((1 - 0.01) / 0.01))
            else: raise NotImplementedError()
            cls_net.append(cls_last)
            cls_net = nn.Sequential(*cls_net)
            self.class_nets.append(cls_net)
        self.n_anch = n_anch
        self.n_cls = n_cls
        self.enable_conf = enable_conf
    
    def forward(self, features: list):
        all_level_preds = []
        for i, x in enumerate(features):
            cls_pred = self.class_nets[i](x)
            bbox_pred = self.bbox_nets[i](x)

            nB, _, nH, nW = bbox_pred.shape
            nA = self.n_anch
            if nA >= 2:
                bbox_pred = bbox_pred.view(nB,nA,-1,nH,nW).permute(0,1,3,4,2)
                cls_pred  = cls_pred.view(nB,nA,-1,nH,nW).permute(0,1,3,4,2)
            elif nA == 1:
                assert bbox_pred.shape[1] == 4
                bbox_pred = bbox_pred.permute(0, 2, 3, 1)
                cls_pred  = cls_pred.permute(0, 2, 3, 1)
            else: raise Exception()
            if self.enable_conf:
                assert cls_pred.shape[-1] == self.n_cls + 1
                raw = {
                    'bbox':  bbox_pred,
                    'conf':  cls_pred[..., 0:1],
                    'class': cls_pred[..., 1:],
                }
            else:
                assert cls_pred.shape[-1] == self.n_cls
                raw = {
                    'bbox':  bbox_pred,
                    'class': cls_pred,
                }
            all_level_preds.append(raw)
        return all_level_preds

def spconv3x3_bn_swish(inout_ch):
    block = nn.Sequential(
        SeparableConv2d(inout_ch, inout_ch, 3, 1, padding=1),
        nn.BatchNorm2d(inout_ch, eps=0.001, momentum=0.01),
        Swish()
    )
    return block


class _LR_TB_last(nn.Module):
    '''
    Custom bounding box head for FCOS
    '''
    def __init__(self, in_ch):
        super().__init__()
        self._lr = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, 2, (1,3), stride=1, padding=(0,1))
        )
        self._tb = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, 2, (3,1), stride=1, padding=(1,0))
        )
    
    def forward(self, x):
        lr = self._lr(x)
        tb = self._tb(x)
        ltrb = torch.stack([
            lr[:,0,:,:], tb[:,0,:,:], lr[:,1,:,:], tb[:,1,:,:]
        ], dim=1)
        return ltrb


class EfDetHead_wCenter(nn.Module):
    '''
    EfficientDet head for FCOS
    '''
    def __init__(self, cfg: dict):
        super().__init__()
        n_cls       = cfg['general.num_class']
        n_anch      = cfg['model.effrpn.num_anchor_per_level']
        feature_chs = cfg['model.fpn.out_channels']
        repeat      = cfg['model.effrpn.repeat_num']
        bb_param    = cfg.get('general.bbox_param', 4)
        enable_conf = cfg.get('model.effrpn.enable_conf', False)
        assert n_anch == 1
        assert cfg['model.effrpn.enable_centerscore']

        self.class_nets  = nn.ModuleList()
        self.bbox_nets   = nn.ModuleList()
        self.bbox_lasts  = nn.ModuleList()
        self.center_nets = nn.ModuleList()
        for ch in feature_chs:
            bb_net = [spconv3x3_bn_swish(ch) for _ in range(repeat)]
            bb_net = nn.Sequential(*bb_net)
            self.bbox_nets.append(bb_net)
            self.bbox_lasts.append(nn.Conv2d(ch, n_anch*bb_param, 3, 1, padding=1))

            ct_net = [
                spconv3x3_bn_swish(ch),
                nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1)
            ]
            ct_net[1].weight.data.normal_(mean=0, std=0.01)
            ct_net[1].bias.data.fill_(-np.log((1 - 0.05) / 0.05))
            self.center_nets.append(nn.Sequential(*ct_net))

            cls_net = [spconv3x3_bn_swish(ch) for _ in range(repeat)]
            # Initialize the final layer bias by -np.log((1 - 0.05) / 0.05)
            # so that the initial confidence are close to 0.05
            if enable_conf:
                cls_last = nn.Conv2d(ch, n_anch * (1+n_cls), 3, 1, padding=1)
            else:
                cls_last = nn.Conv2d(ch, n_anch * n_cls, 3, 1, padding=1)
            cls_last.weight.data.normal_(mean=0, std=0.01)
            cls_last.bias.data.fill_(-np.log((1 - 0.05) / 0.05))
            cls_net.append(cls_last)
            cls_net = nn.Sequential(*cls_net)
            self.class_nets.append(cls_net)
        # self.n_anch = n_anch
        self.n_cls = n_cls
        self.enable_conf = enable_conf
    
    def forward(self, features: list):
        all_level_preds = []
        for i, x in enumerate(features):
            cls_pred    = self.class_nets[i](x)
            bbox_feats  = self.bbox_nets[i](x)
            bbox_pred   = self.bbox_lasts[i](bbox_feats)
            center_pred = self.center_nets[i](bbox_feats)

            nB, _, nH, nW = bbox_pred.shape
            assert bbox_pred.shape[1] == 4
            # nA = self.n_anch
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            center_pred = center_pred.permute(0, 2, 3, 1)
            cls_pred = cls_pred.permute(0, 2, 3, 1)

            if self.enable_conf:
                assert cls_pred.shape[-1] == self.n_cls + 1
                raw = {
                    'bbox': bbox_pred,
                    'center': center_pred,
                    'conf': cls_pred[..., 0:1],
                    'class': cls_pred[..., 1:],
                }
            else:
                assert cls_pred.shape[-1] == self.n_cls
                raw = {
                    'bbox': bbox_pred,
                    'center': center_pred,
                    'class': cls_pred,
                }
            all_level_preds.append(raw)
        return all_level_preds
