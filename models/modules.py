import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    '''
    Depthwise separable convolution 2D
    '''
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
    Residual block in Darknet53

    Args:
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


class ConvBnLeaky(nn.Module):
    '''
    Standard Conv2d + BatchNorm + LeakyReLU

    Args:
        c1: input channel, e.g. 32
        c2: output channel, e.g. 64
        k: kernel size, e.g. 3 or (3,3)
        s: stride, e.g. 1 or (1,1)
        g: groups
    '''
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-5, momentum=0.01)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
