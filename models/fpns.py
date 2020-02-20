import torch
import torch.nn as nn
import torch.nn.functional as tnf

from .backbones import ConvBnLeaky



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
    def __init__(self, in_channels=(256, 512, 1024), class_num=80):
        super().__init__()
        ch3, ch4, ch5 = in_channels
        out_ch = (class_num + 5) * 3
        self.branch_P3 = YOLOBranch(ch3, out_ch, prev_ch=(ch4//2,ch3//2))
        self.branch_P4 = YOLOBranch(ch4, out_ch, prev_ch=(ch5//2,ch4//2))
        self.branch_P5 = YOLOBranch(ch5, out_ch)
    
    def forward(self, features):
        p3, p4, p5 = features
        # go through FPN blocks in three scales
        p5_fpn, p5_to_p4 = self.branch_P5(p5, previous=None)
        p4_fpn, p4_to_p3 = self.branch_P4(p4, previous=p5_to_p4)
        p3_fpn, _ = self.branch_P3(p3, previous=p4_to_p3)

        return [p3_fpn, p4_fpn, p5_fpn]
