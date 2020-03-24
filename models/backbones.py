import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from external_packages.efficientnet.model import EfficientNet
from external_packages.maskrcnn_benchmark_resnet import ResNet


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


class EfficientNetBackbone(EfficientNet):
    def forward(self, inputs):
        """
        Serve as a backbone for YOLOv3. Returns output of the last three scales
        """
        ini_size = inputs.shape[-1]
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        features = []
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            y = block(x, drop_connect_rate=drop_connect_rate)
            if y.shape[-1] != x.shape[-1]:
                features.append(x)
            x = y

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        features.append(x)

        # extract features for YOLOv3
        small, medium, large = features[-3], features[-2], features[-1]
        assert ini_size/small.shape[-1] == 8 and ini_size/large.shape[-1] == 32
        return small, medium, large

def efficientnet(model_name):
    print(f'Using backbone {model_name}. Loading ImageNet weights...')
    model = EfficientNetBackbone.from_pretrained(model_name, advprop=True)
    return model

def efficient_feature_info(model_name):
    feature_dict = {
        # Coefficients:   (small, medium, large)
        'efficientnet-b0': (40, 112, 1280),
        'efficientnet-b1': (40, 112, 1280),
        'efficientnet-b2': (48, 120, 1408),
        'efficientnet-b3': (48, 136, 1536),
        'efficientnet-b4': (56, 160, 1792),
        'efficientnet-b5': (64, 176, 2048),
        'efficientnet-b6': (72, 200, 2304),
        'efficientnet-b7': (80, 224, 2560),
        'efficientnet-b8': (88, 248, 2816)
    }
    return feature_dict[model_name]
