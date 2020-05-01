import torch
import torch.nn as nn
import torchvision.models

from external_packages.efficientnet.model import EfficientNet, MemoryEfficientSwish
from .modules import SeparableConv2d


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
        nn.BatchNorm2d(out_, eps=1e-5, momentum=0.01),
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
    def __init__(self, cfg: dict):
        super().__init__()
        model_name = cfg['model.backbone.name']
        assert model_name in self.valid_names, 'Unknown efficientnet model name'
        efn = EfficientNet.from_pretrained(model_name, advprop=True)
        del efn._conv_head, efn._bn1, efn._avg_pooling, efn._dropout, efn._fc
        self.model = efn
        
        efnet_chs = [efn._blocks_args[i].output_filters for i in [2,4,6]]
        if cfg['model.backbone.num_levels'] == 3:
            self.feature_chs = efnet_chs
            self.feature_strides = (8, 16, 32)
            self.C6C7 = False
        elif cfg['model.backbone.num_levels'] == 5:
            out_ch = cfg['model.backbone.C6C7_out_channels']
            downsample_method = cfg.get('model.efficientnet.C6C7_downsample', 'maxpool')
            if downsample_method == 'maxpool':
                self.c5_to_c6 = nn.Sequential(
                    nn.Conv2d(efnet_chs[-1], out_ch, 1, stride=1, padding=0),
                    nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.01),
                    nn.MaxPool2d(3, stride=2, padding=1)
                )
                self.c6_to_c7 = nn.MaxPool2d(3, stride=2, padding=1)
            elif downsample_method == 'spconv':
                self.c5_to_c6 = nn.Sequential(
                    SeparableConv2d(efnet_chs[-1], out_ch, 3, stride=1, padding=1),
                    nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.01),
                    nn.MaxPool2d(3, stride=2, padding=1)
                )
                self.c6_to_c7 = nn.Sequential(
                    SeparableConv2d(out_ch, out_ch, 3, stride=1, padding=1),
                    nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.01),
                    nn.MaxPool2d(3, stride=2, padding=1)
                )
            else: raise NotImplementedError()
            self.feature_chs = efnet_chs + [out_ch, out_ch]
            self.feature_strides = (8, 16, 32, 64, 128)
            self.C6C7 = True
        else:
            raise NotImplementedError()
        self.enable_dropout = cfg['model.efficientnet.enable_dropout']

    def forward(self, x):
        x = self.model._swish(self.model._bn0(self.model._conv_stem(x)))
        features = []
        for idx, block in enumerate(self.model._blocks):
            # print(block._depthwise_conv.stride)
            if self.enable_dropout:
                drop_connect_rate = self.model._global_params.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self.model._blocks)
            else:
                drop_connect_rate = None
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
