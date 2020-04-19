import torch
import torch.nn as nn
import torch.nn.functional as tnf

from .backbones import ConvBnLeaky
from .modules import SeparableConv2d, MemoryEfficientSwish


class YOLOBranch(nn.Module):
    '''
    Args:
        in_: int, input channel number
        out_: int, output channel number, typically = 3 * 6 [x,y,w,h,a,conf]
        has_previous: bool, True if this is not the first detection layer
        prev_ch: (int,int), the Conv2d channel for the previous feature,
                 default: None
    '''
    def __init__(self, in_, prev_ch=None):
        super(YOLOBranch, self).__init__()
        assert in_ % 2 == 0, 'input channel must be divisible by 2'

        # tmp_ch = prev_ch if prev_ch is not None else (in_, in_//2)
        if prev_ch:
            self.process = ConvBnLeaky(prev_ch[0], prev_ch[1], k=1, s=1)
            in_after_cat = in_ + prev_ch[1]
        else:
            in_after_cat = in_

        self.cbl_0 = ConvBnLeaky(in_after_cat, in_//2, k=1, s=1)
        self.cbl_1 = ConvBnLeaky(in_//2, in_, k=3, s=1)

        self.cbl_2 = ConvBnLeaky(in_, in_//2, k=1, s=1)
        self.cbl_3 = ConvBnLeaky(in_//2, in_, k=3, s=1)

        self.cbl_4 = ConvBnLeaky(in_, in_//2, k=1, s=1)
        self.cbl_5 = ConvBnLeaky(in_//2, in_, k=3, s=1)
        
    def forward(self, x, previous=None):
        '''
        x: feature from backbone, for large/medium/small size
        previous: feature from previous yolo layer
        '''
        if previous is not None:
            pre = self.process(previous)
            pre = tnf.interpolate(pre, size=x.shape[2:4], mode='nearest')
            x = torch.cat((pre, x), dim=1)
        
        x = self.cbl_0(x)
        x = self.cbl_1(x)
        x = self.cbl_2(x)
        x = self.cbl_3(x)
        feature = self.cbl_4(x)
        x = self.cbl_5(feature)

        return x, feature


class YOLOv3FPN(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        assert cfg['model.backbone.num_levels'] == 3
        ch3, ch4, ch5 = cfg['model.backbone.out_channels']
        self.branch_P3 = YOLOBranch(ch3, prev_ch=(ch4//2,ch3//2))
        self.branch_P4 = YOLOBranch(ch4, prev_ch=(ch5//2,ch4//2))
        self.branch_P5 = YOLOBranch(ch5)
    
    def forward(self, features):
        c3, c4, c5 = features
        # go through FPN blocks in three scales
        p5, c5_to_c4 = self.branch_P5(c5, previous=None)
        p4, c4_to_c3 = self.branch_P4(c4, previous=c5_to_c4)
        p3, _ = self.branch_P3(c3, previous=c4_to_c3)

        return [p3, p4, p5]


# class LastLevelP6P7(nn.Module):
#     """
#     This module is used in RetinaNet to generate extra layers, P6 and P7.
#     """
#     def __init__(self, in_channels, out_channels):
#         super(LastLevelP6P7, self).__init__()
#         self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
#         self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
#         for module in [self.p6, self.p7]:
#             nn.init.kaiming_uniform_(module.weight, a=1)
#             nn.init.constant_(module.bias, 0)
#         self.use_P5 = in_channels == out_channels

#     def forward(self, c5, p5):
#         x = p5 if self.use_P5 else c5
#         p6 = self.p6(x)
#         p7 = self.p7(tnf.relu(p6))
#         return [p6, p7]

class C3toP5FPN(nn.Module):
    '''
    Args:
        in_channels: list/tuple of int
    '''
    def __init__(self, in_channels=(512, 1024, 2048), out_ch=256):
        super().__init__()
        assert len(in_channels) == 3

        ch3, ch4, ch5 = in_channels
        self.c5_to_p5_ = conv_uniform(ch5, out_ch, kernel_size=1)
        self.p5_to_p5 = conv_uniform(out_ch, out_ch, kernel_size=3, stride=1)
        self.c4_to_p4_ = conv_uniform(ch4, out_ch, kernel_size=1)
        self.p4_to_p4 = conv_uniform(out_ch, out_ch, kernel_size=3, stride=1)
        self.c3_to_p3_ = conv_uniform(ch3, out_ch, kernel_size=1)
        self.p3_to_p3 = conv_uniform(out_ch, out_ch, kernel_size=3, stride=1)

        self.p5_to_p6 = nn.Conv2d(out_ch, out_ch, 3, 2, 1)
        self.p6_to_p7 = nn.Conv2d(out_ch, out_ch, 3, 2, 1)

    def forward(self, features):
        C3, C4, C5 = features
        results = []

        p5_ = self.c5_to_p5_(C5)
        p5 = self.p5_to_p5(p5_)

        top_down = tnf.interpolate(p5_, C4.shape[-2:], mode='nearest')
        p4_ = self.c4_to_p4_(C4) + top_down
        p4 = self.p4_to_p4(p4_)
        
        top_down = tnf.interpolate(p4_, C3.shape[-2:], mode='nearest')
        p3_ = self.c3_to_p3_(C3) + top_down
        p3 = self.p3_to_p3(p3_)

        p6 = self.p5_to_p6(p5_)
        p7 = self.p6_to_p7(tnf.relu(p6))

        return (p3, p4, p5, p6, p7)


# Source: https://github.com/facebookresearch/maskrcnn-benchmark
class RetinaNetFPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """
    def __init__(self, feature_channels, out_channels=256):
    #     self, in_channels_list, out_channels, top_blocks=None
    # ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super().__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(feature_channels):
            # use idx + 2 just to load the official weights
            inner_block = "fpn_inner{}".format(idx+2)
            layer_block = "fpn_layer{}".format(idx+2)

            if in_channels == 0:
                continue
            inner_block_module = conv_uniform(in_channels, out_channels, 1)
            layer_block_module = conv_uniform(out_channels, out_channels, 3, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = LastLevelP6P7(out_channels, out_channels)

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            # inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            inner_top_down = tnf.interpolate(last_inner, 
                size=(int(inner_lateral.shape[-2]), int(inner_lateral.shape[-1])),
                mode='nearest'
            )
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        else:
            raise Exception()
        # elif isinstance(self.top_blocks, LastLevelMaxPool):
        #     last_results = self.top_blocks(results[-1])
        #     results.extend(last_results)

        return tuple(results)


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(tnf.relu(p6))
        return [p6, p7]


def conv_uniform(in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 use_gn=False, use_relu=False):
    conv = nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=dilation * (kernel_size - 1) // 2, 
        dilation=dilation, 
        bias=False if use_gn else True
    )
    # Caffe2 implementation uses XavierFill, which in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(conv.weight, a=1)
    if not use_gn:
        nn.init.constant_(conv.bias, 0)
    module = [conv,]
    if use_gn:
        raise NotImplementedError()
        # module.append(group_norm(out_channels))
    if use_relu:
        module.append(nn.ReLU(inplace=True))
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv


# def get_bifpn(in_channels, out_ch, repeat_num, fusion_method='linear'):
def get_bifpn(cfg: dict):
    in_channels = cfg['model.backbone.out_channels']
    out_ch = cfg['model.bifpn.out_ch']
    repeat_num = cfg['model.bifpn.repeat_num']
    fusion_method = cfg['model.bifpn.fusion_method']

    assert repeat_num >= 1
    if len(in_channels) == 3:
        fpn_func = BiFPN3
    elif len(in_channels) == 5:
        fpn_func = BiFPN5
    else:
        raise NotImplementedError()
    fpn = []
    fpn.append(fpn_func(out_ch, fusion_method=fusion_method, in_chs=in_channels))
    for _ in range(repeat_num-1):
        fpn.append(fpn_func(out_ch, fusion_method=fusion_method))
    fpn = nn.Sequential(*fpn)
    return fpn


class BiFPN3(nn.Module):
    def __init__(self, fpn_ch, fusion_method='linear', in_chs=None):
        super().__init__()
        if in_chs: assert len(in_chs) == 3
        self.p3in_out = conv1x1_bn(in_chs[0], fpn_ch) if in_chs else lambda x:x
        self.p4in_m = conv1x1_bn(in_chs[1], fpn_ch) if in_chs else lambda x:x
        self.p4in_out = conv1x1_bn(in_chs[1], fpn_ch) if in_chs else lambda x:x
        self.p5in_4m = conv1x1_bn(in_chs[2], fpn_ch) if in_chs else lambda x:x
        self.p5in_out = conv1x1_bn(in_chs[2], fpn_ch) if in_chs else lambda x:x

        if fusion_method == 'linear':
            fusion_layer = LinearFusion
        elif fusion_method == 'softmax':
            raise NotImplementedError()
        self.fuse_4m = fusion_layer(num=2, channels=fpn_ch)
        self.fuse_3out = fusion_layer(num=2, channels=fpn_ch)
        self.fuse_4out = fusion_layer(num=3, channels=fpn_ch)
        self.fuse_5out = fusion_layer(num=2, channels=fpn_ch)

    def forward(self, features):
        """
            P5in ------------------------- P5out -------->
                 ------>-----|               |
            P4in ---------- P4m ---------- P4out -------->
                             |------>-----   |
            P3in ------------------------- P3out -------->
        """
        P3in, P4in, P5in = features
        assert P3in.shape[2] == P4in.shape[2]*2 == P5in.shape[2]*4
        # P4in + P5in -> P4m
        P4m = self.fuse_4m(self.p4in_m(P4in), upsample2x(self.p5in_4m(P5in)))
        # P3in + P4m -> P3out
        P3out = self.fuse_3out(self.p3in_out(P3in), upsample2x(P4m))
        # P4in + P4m + P3out -> P4out
        _P3out_to_P4 = tnf.max_pool2d(P3out, kernel_size=3, stride=2, padding=1)
        P4out = self.fuse_4out(self.p4in_out(P4in), P4m, _P3out_to_P4)
        # P5in + P4out -> P5out
        _P4out_to_P5 = tnf.max_pool2d(P4out, kernel_size=3, stride=2, padding=1)
        P5out = self.fuse_5out(self.p5in_out(P5in), _P4out_to_P5)
        return [P3out, P4out, P5out]


class BiFPN5(nn.Module):
    def __init__(self, fpn_ch, fusion_method='linear', in_chs=None):
        super().__init__()
        if in_chs:
            assert len(in_chs) == 5
            assert in_chs[3] == fpn_ch and in_chs[4] == fpn_ch
        self.p3in_out = conv1x1_bn(in_chs[0], fpn_ch) if in_chs else lambda x:x
        self.p4in_m = conv1x1_bn(in_chs[1], fpn_ch) if in_chs else lambda x:x
        self.p4in_out = conv1x1_bn(in_chs[1], fpn_ch) if in_chs else lambda x:x
        self.p5in_m = conv1x1_bn(in_chs[2], fpn_ch) if in_chs else lambda x:x
        self.p5in_out = conv1x1_bn(in_chs[2], fpn_ch) if in_chs else lambda x:x
        # self.in_chs = in_chs

        if fusion_method == 'linear':
            fusion_layer = LinearFusion
        elif fusion_method == 'softmax':
            raise NotImplementedError()
        self.fuse_6m = fusion_layer(num=2, channels=fpn_ch)
        self.fuse_5m = fusion_layer(num=2, channels=fpn_ch)
        self.fuse_4m = fusion_layer(num=2, channels=fpn_ch)
        self.fuse_3out = fusion_layer(num=2, channels=fpn_ch)
        self.fuse_4out = fusion_layer(num=3, channels=fpn_ch)
        self.fuse_5out = fusion_layer(num=3, channels=fpn_ch)
        self.fuse_6out = fusion_layer(num=3, channels=fpn_ch)
        self.fuse_7out = fusion_layer(num=2, channels=fpn_ch)

    def forward(self, features):
        """
            P7in ------------------------- P7out -------->
                 ------>-----|               |
            P6in ---------- P6m ---------- P6out -------->
                             |               |
            P5in ---------- P5m ---------- P5out -------->
                             |               |
            P4in ---------- P4m ---------- P4out -------->
                             |------>-----   |
            P3in ------------------------- P3out -------->
        """
        P3in, P4in, P5in, P6in, P7in = features
        assert P3in.shape[2] == P4in.shape[2]*2 == P5in.shape[2]*4 \
               == P6in.shape[2]*8 == P7in.shape[2]*16
        # P6in + P7in -> P6m
        P6m = self.fuse_6m(P6in, upsample2x(P7in))
        # P5in + P6m -> P5m
        P5m = self.fuse_5m(self.p5in_m(P5in), upsample2x(P6m))
        # P4in + P5m -> P4m
        P4m = self.fuse_4m(self.p4in_m(P4in), upsample2x(P5m))
        # P3in + P4m -> P3out
        P3out = self.fuse_3out(self.p3in_out(P3in), upsample2x(P4m))
        # P4in + P4m + P3out -> P4out
        _P3out_to_P4 = tnf.max_pool2d(P3out, kernel_size=3, stride=2, padding=1)
        P4out = self.fuse_4out(self.p4in_out(P4in), P4m, _P3out_to_P4)
        # P5in + P5m + P4out -> P5out
        _P4out_to_P5 = tnf.max_pool2d(P4out, kernel_size=3, stride=2, padding=1)
        P5out = self.fuse_5out(self.p5in_out(P5in), P5m, _P4out_to_P5)
        # P6in + P6m + P5out -> P6out
        _P5out_to_P6 = tnf.max_pool2d(P5out, kernel_size=3, stride=2, padding=1)
        P6out = self.fuse_6out(P6in, P6m, _P5out_to_P6)
        # P7in + P6out -> P7out
        _P6out_to_P7 = tnf.max_pool2d(P6out, kernel_size=3, stride=2, padding=1)
        P7out = self.fuse_7out(P7in, _P6out_to_P7)
        return [P3out, P4out, P5out, P6out, P7out]


class LinearFusion(nn.Module):
    def __init__(self, num, channels):
        super().__init__()
        self.num = num
        self.weights = nn.Parameter(torch.ones(num), requires_grad=True)
        self.spconv_bn = nn.Sequential(
            SeparableConv2d(channels, channels, 3, 1, padding=1),
            nn.BatchNorm2d(channels, eps=0.001, momentum=0.01)
        )
        self.swish = MemoryEfficientSwish()
    
    def forward(self, *features):
        assert isinstance(features, (list,tuple)) and len(features) == self.num
        weights = tnf.relu(self.weights)
        weighted = [w*x for w,x in zip(weights, features)]
        sum_ = weights.sum() + 0.0001
        fused = sum(weighted) / sum_
        fused = self.spconv_bn(self.swish(fused))
        return fused

def upsample2x(x):
    return tnf.interpolate(x, scale_factor=(2,2), mode='nearest')

# def swish(x):
#     return x * torch.sigmoid(x)

def conv1x1_bn(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0),
        nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.01),
    )
