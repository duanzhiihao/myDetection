import torch
import torch.nn as nn
import torch.nn.functional as tnf

from .backbones import ConvBnLeaky


def get_fpn(name, backbone_info=None, **kwargs):
    if name == 'yolo3':
        return YOLOv3FPN(anch_num=3, **kwargs)
    elif name == 'yolo3_1anch':
        return YOLOv3FPN(anch_num=1, **kwargs)
    elif name == 'retina':
        rpn = RetinaNetFPN(backbone_info['feature_channels'], 256)
        info = {
            'strides'
        }
        return rpn, 
    else:
        raise Exception('Unknown FPN name')


class YOLOBranch(nn.Module):
    '''
    Args:
        in_: int, input channel number
        out_: int, output channel number, typically = 3 * 6 [x,y,w,h,a,conf]
        has_previous: bool, True if this is not the first detection layer
        prev_ch: (int,int), the Conv2d channel for the previous feature,
                 default: None
    '''
    # def __init__(self, in_, out_=18, has_previous=False, prev_ch=None):
    def __init__(self, in_, out_=18, prev_ch=None):
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

        self.to_box = nn.Conv2d(in_, out_, kernel_size=1, stride=1)
        
    def forward(self, x, previous=None):
        '''
        x: feature from backbone, for large/medium/small size
        previous: feature from previous yolo layer
        '''
        if previous is not None:
            pre = self.process(previous)
            pre = tnf.interpolate(pre, scale_factor=2, mode='nearest')
            x = torch.cat((pre, x), dim=1)
        
        x = self.cbl_0(x)
        x = self.cbl_1(x)
        x = self.cbl_2(x)
        x = self.cbl_3(x)
        feature = self.cbl_4(x)
        x = self.cbl_5(feature)
        detection = self.to_box(x)

        return detection, feature


class YOLOv3FPN(nn.Module):
    def __init__(self, in_channels=(256, 512, 1024), class_num=80, anch_num=3):
        super().__init__()
        ch3, ch4, ch5 = in_channels
        out_ch = (class_num + 5) * anch_num
        self.branch_P3 = YOLOBranch(ch3, out_ch, prev_ch=(ch4//2,ch3//2))
        self.branch_P4 = YOLOBranch(ch4, out_ch, prev_ch=(ch5//2,ch4//2))
        self.branch_P5 = YOLOBranch(ch5, out_ch)
    
    def forward(self, features):
        c3, c4, c5 = features
        # go through FPN blocks in three scales
        p5, c5_to_c4 = self.branch_P5(c5, previous=None)
        p4, c4_to_c3 = self.branch_P4(c4, previous=c5_to_c4)
        p3, _ = self.branch_P3(c3, previous=c4_to_c3)

        return [p3, p4, p5]


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
