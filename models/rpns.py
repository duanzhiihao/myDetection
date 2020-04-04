import torch.nn as nn

from .modules import SeparableConv2d, Swish


def get_rpn(name, chs, **kwargs):
    if name == 'yv3':
        return YOLOHead(in_channels=chs, **kwargs)
    else:
        raise NotImplementedError()


class YOLOHead(nn.Module):
    def __init__(self, in_channels=(256, 512, 1024), **kwargs):
        super().__init__()
        n_anch = kwargs.get('num_anchor_per_level', 3)
        n_cls = kwargs.get('num_class', 80)
        self.heads = nn.ModuleList()
        out_ch = (n_cls + 5) * n_anch
        for ch in in_channels:
            self.heads.append(nn.Conv2d(ch, out_ch, 1, stride=1))
        self.n_anch = n_anch
        self.n_cls = n_cls

    def forward(self, features):
        all_level_preds = []
        for module, P in zip(self.heads, features):
            preds = module(P)
            nB, _, nH, nW = preds.shape
            preds = preds.view(nB, self.n_anch, self.n_cls+5, nH, nW)
            
            raw = {
                'bbox': preds[:, :, 0:4, :, :].permute(0, 1, 3, 4, 2),
                'conf': preds[:, :, 4:5, :, :].permute(0, 1, 3, 4, 2),
                'class': preds[:, :, 5:, :, :].permute(0, 1, 3, 4, 2),
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
        self.cls_tower = nn.Sequential(*cls_tower)
        self.cls_pred = nn.Conv2d(in_ch, num_class, 3, stride=1, padding=1)
        self.bbox_tower = nn.Sequential(*bbox_tower)
        self.bbox_pred = nn.Conv2d(in_ch, 4, 3, stride=1, padding=1)
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
            cls_pred = self.cls_pred(cls_tower)
            # bbox_reg.append(bbox_pred)

            raw = {
                'bbox': bbox_pred,
                'class': cls_pred,
                'center': conf_pred,
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
    def __init__(self, feature_chs, repeat, cls_ch=720, bbox_ch=36):
        super().__init__()
        self.class_nets = nn.ModuleList()
        self.bbox_nets = nn.ModuleList()
        for ch in feature_chs:
            cls_net = [spconv3x3_bn_swish(ch) for _ in range(repeat)]
            cls_net.append(SeparableConv2d(ch, cls_ch, 3, 1, padding=1))
            cls_net = nn.Sequential(*cls_net)
            self.class_nets.append(cls_net)
            
            bb_net = [spconv3x3_bn_swish(ch) for _ in range(repeat)]
            bb_net.append(SeparableConv2d(ch, bbox_ch, 3, 1, padding=1))
            bb_net = nn.Sequential(*bb_net)
            self.bbox_nets.append(bb_net)
    
    def forward(self, features: list):
        all_level_preds = []
        for i, x in enumerate(features):
            cls_pred = self.class_nets[i](x)
            bbox_pred = self.bbox_nets[i](x)
            raw = {
                'bbox': bbox_pred,
                'class': cls_pred,
            }
            all_level_preds.append(raw)
        return all_level_preds

def spconv3x3_bn_swish(inout_ch):
    return nn.Sequential(
        SeparableConv2d(inout_ch, inout_ch, 3, 1, padding=1),
        nn.BatchNorm2d(inout_ch, eps=0.001, momentum=0.99),
        Swish()
    )
