import torch
import torch.nn as nn
import torchvision.models

from external_packages.efficientnet.model import EfficientNet
from .modules import Swish

# def get_backbone(name):
#     if name == 'dark53':
#         backbone = Darknet53()
#         print("Using backbone Darknet-53. Loading ImageNet weights....")
#         pretrained = torch.load('./weights/dark53_imgnet.pth')
#         backbone.load_state_dict(pretrained)
#         return backbone, (256, 512, 1024), (8, 16, 32)
#     # elif name == 'res34':
#     #     self.backbone = models.backbones.resnet34()
#     elif name == 'res50':
#         backbone = ResNet('res50')
#         info = {
#             'feature_channels': (512, 1024, 2048),
#             'feature_strides': (8, 16, 32),
#         }
#         return backbone, info
#     # elif name == 'res101':
#     #     self.backbone = models.backbones.resnet101()
#     # elif 'efficientnet' in backbone:
#     #     self.backbone = models.backbones.efficientnet(backbone)
#     else:
#         raise Exception('Unknown backbone name')


def get_backbone_fpn(name):
    if name == 'res50_retina':
        from external_packages.maskrcnn_benchmark_resnet import ResNet
        backbone = ResNet('res50')
        from .fpns import RetinaNetFPN
        fpn = RetinaNetFPN(feature_channels=(512, 1024, 2048), out_channels=256)
        info = {
            'feature_channels': 256,
            'feature_strides': (8, 16, 32, 64, 128),
        }
    elif name == 'my_res50_retina':
        backbone = get_resnet50()
        from .fpns import C3toP5FPN
        fpn = C3toP5FPN(in_channels=(512, 1024, 2048), out_ch=256)
        info = {
            'feature_channels': 256,
            'feature_strides': (8, 16, 32, 64, 128),
        }
    elif name == 'dark53_yv3':
        backbone = Darknet53()
        from .fpns import YOLOv3FPN
        fpn = YOLOv3FPN(in_channels=(256, 512, 1024))
        info = {
            'feature_channels': (256, 512, 1024),
            'feature_strides': (8, 16, 32),
        }
    elif name == 'b0_yv3_345':
        backbone = EfNetBackbone(model_name='efficientnet-b0', C6C7=False)
        from .fpns import YOLOv3FPN
        fpn = YOLOv3FPN(in_channels=backbone.feature_chs)
        info = {
            'feature_channels': backbone.feature_chs,
            'feature_strides': (8, 16, 32),
        }
    elif name in {'d0_345', 'd1_345', 'd2_345', 'd3_345', 'd4_345'}:
        id2info = {
            'd0': (64, 3), 'd1': (88, 4), 'd2': (112, 5), 'd3': (160, 6),
            'd4': (224, 7),
        }
        backbone = EfNetBackbone('efficientnet-b'+name[1], C6C7=False)
        from .fpns import get_bifpn
        fpn_ch, fpn_num = id2info[name[:2]]
        fpn = get_bifpn(backbone.feature_chs, out_ch=fpn_ch, repeat_num=fpn_num)
        info = {
            'feature_channels': (fpn_ch, fpn_ch, fpn_ch),
            'feature_strides': (8, 16, 32),
        }
    else:
        raise NotImplementedError()
    
    return backbone, fpn, info



def ConvBnLeaky(in_, out_, k, s):
    '''
    in_: input channel, e.g. 32
    out_: output channel, e.g. 64
    k: kernel size, e.g. 3 or (3,3)
    s: stride, e.g. 1 or (1,1)
    '''
    pad = (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_, out_, k, s, padding=pad, bias=False),
        nn.BatchNorm2d(out_, eps=1e-5, momentum=0.9),
        nn.LeakyReLU(0.1)
    )


class DarkBlock(nn.Module):
    '''
    basic residual block in Darknet53
    in_out: input and output channels
    hidden: channels in the block
    '''
    def __init__(self, in_out, hidden):
        super().__init__()
        self.cbl_0 = ConvBnLeaky(in_out, hidden, k=1, s=1)
        self.cbl_1 = ConvBnLeaky(hidden, in_out, k=3, s=1)

    def forward(self, x):
        residual = x
        x = self.cbl_0(x)
        x = self.cbl_1(x)

        return x + residual


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.netlist = nn.ModuleList()
        
        # first conv layer
        self.netlist.append(ConvBnLeaky(3, 32, k=3, s=1))

        # Downsampled by 2 (accumulatively), followed by residual blocks
        self.netlist.append(ConvBnLeaky(32, 64, k=3, s=2))
        for _ in range(1):
            self.netlist.append(DarkBlock(in_out=64, hidden=32))

        # Downsampled by 4 (accumulatively), followed by residual blocks
        self.netlist.append(ConvBnLeaky(64, 128, k=3, s=2))
        for _ in range(2):
            self.netlist.append(DarkBlock(in_out=128, hidden=64))
        
        # Downsampled by 8 (accumulatively), followed by residual blocks
        self.netlist.append(ConvBnLeaky(128, 256, k=3, s=2))
        for _ in range(8):
            self.netlist.append(DarkBlock(in_out=256, hidden=128))
        assert len(self.netlist) == 15

        # Downsampled by 16 (accumulatively), followed by residual blocks
        self.netlist.append(ConvBnLeaky(256, 512, k=3, s=2))
        for _ in range(8):
            self.netlist.append(DarkBlock(in_out=512, hidden=256))
        assert len(self.netlist) == 24

        # Downsampled by 32 (accumulatively), followed by residual blocks
        self.netlist.append(ConvBnLeaky(512, 1024, k=3, s=2))
        for _ in range(4):
            self.netlist.append(DarkBlock(in_out=1024, hidden=512))
        assert len(self.netlist) == 29
        # end creating Darknet-53 back bone layers

    def forward(self, x):
        for i in range(0,15):
            x = self.netlist[i](x)
        C3 = x
        for i in range(15,24):
            x = self.netlist[i](x)
        C4 = x
        for i in range(24,29):
            x = self.netlist[i](x)
        C5 = x
        # We expect that C3 contains information about small objects,
        # and C5 contains information about large objects
        return [C3, C4, C5]


class ResNetBackbone(nn.Module):
    '''
    Args:
        tv_model: torch vision model
    '''
    def __init__(self, tv_model):
        super().__init__()
        self.conv1 = tv_model.conv1
        self.bn1 = tv_model.bn1
        self.relu = tv_model.relu
        self.maxpool = tv_model.maxpool

        self.layer1 = tv_model.layer1
        self.layer2 = tv_model.layer2
        self.layer3 = tv_model.layer3
        self.layer4 = tv_model.layer4
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        C3 = self.layer2(x)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)
        return [C3, C4, C5]


def get_resnet50():
    print('Using backbone ResNet-50. Loading ImageNet weights...')
    model = torchvision.models.resnet50(pretrained=True)
    return ResNetBackbone(model)


class EfNetBackbone(nn.Module):
    '''
    Args:
        model_name: str, e.g., 'efficientnet-b0'
        out_ch: int, e.g., 64
    '''
    valid_names = {
        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
        'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
        'efficientnet-b6'
    }
    def __init__(self, model_name, out_ch=None, C6C7=True):
        super().__init__()
        assert model_name in self.valid_names, 'Unknown efficientnet model name'
        efn = EfficientNet.from_pretrained(model_name, advprop=True)
        del efn._conv_head, efn._bn1, efn._avg_pooling, efn._dropout, efn._fc
        self.model = efn
        efnet_chs = [efn._blocks_args[i].output_filters for i in [2,4,6]]
        if C6C7:
            self.c5_to_c6 = conv1x1_bn_relu_maxp(efnet_chs[-1], out_ch)
            self.c6_to_c7 = conv1x1_bn_relu_maxp(out_ch, out_ch)
            self.feature_chs = efnet_chs + [out_ch, out_ch]
        else:
            self.feature_chs = efnet_chs
        self.C6C7 = C6C7

    def forward(self, x):
        x = self.model._swish(self.model._bn0(self.model._conv_stem(x)))
        features = []
        for idx, block in enumerate(self.model._blocks):
            # print(block._depthwise_conv.stride)
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            y = block(x, drop_connect_rate=drop_connect_rate)
            if y.shape[-1] != x.shape[-1]:
                features.append(x)
            x = y
        features.append(x)
        # C1 (2x), C2 (4x), C3 (8x), C4 (16x), C5 (32x)
        C1, C2, C3, C4, C5 = features
        if self.C6C7:
            C6 = self.c5_to_c6(C5)
            C7 = self.c6_to_c7(C6)
            return [C3, C4, C5, C6, C7]
        else:
            return [C3, C4, C5]


def conv1x1_bn_relu_maxp(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0),
        nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.99),
        Swish(),
        nn.MaxPool2d(3, stride=2, padding=1)
    )