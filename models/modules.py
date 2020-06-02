import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride,
                                   padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, padding=0)
        nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.pointwise.weight, nonlinearity='relu')
        self.pointwise.bias.data.zero_()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# class SwishImplementation(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, i):
#         result = i * torch.sigmoid(i)
#         ctx.save_for_backward(i)
#         return result

#     @staticmethod
#     def backward(ctx, grad_output):
#         i = ctx.saved_variables[0]
#         sigmoid_i = torch.sigmoid(i)
#         return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

# class MemoryEfficientSwish(nn.Module):
#     def forward(self, x):
#         return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# def custom_init(m):
#     classname = m.__class__.__name__
#     if isinstance(m, nn.Conv2d):
#         torch.nn.init.normal_(m.weight.data, 0, 0.01)
#         if m.bias is not None:
#             m.bias.data.zeros_()
#     elif 'BatchNorm' in classname:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         m.bias.data.zeros_()


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
        nn.LeakyReLU(0.1, inplace=True)
    )


class UConvBnLeaky(nn.Module):
    '''
    Standard Conv2d + BatchNorm + LeakyReLU

    Args:
        ch_in, ch_out, kernel, stride, groups
    '''
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    '''
    Focus wh information into c-space
    '''
    def __init__(self, c1, c2, k=1):
        super(Focus, self).__init__()
        self.conv = UConvBnLeaky(c1*4, c2, k=k, s=1)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([
            x[..., ::2, ::2], x[..., 1::2, ::2],
            x[..., ::2, 1::2], x[..., 1::2, 1::2]
        ], 1))


class UBottleneck(nn.Module):
    '''
    ch_in, ch_out, shortcut, groups, expansion
    '''
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = UConvBnLeaky(c1, c_, 1, 1)
        self.cv2 = UConvBnLeaky(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPP(nn.Module):  # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = UConvBnLeaky(c1, c_, 1, 1)
        self.cv2 = UConvBnLeaky(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(x, 1, padding=x//2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
