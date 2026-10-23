"""Drop-in replacement for DCNv2's `DCN` layer (CharlesShang/DCNv2) used by AECR-Net.

The original needs a compiled CUDA extension (Linux only). torchvision ships the same
modulated deformable convolution (DCNv2) as `torchvision.ops.deform_conv2d`, which runs on
CPU and GPU on Windows. Parameter names (weight, bias, conv_offset_mask.*) are identical,
so the official AECR-Net checkpoints load without renaming. Offset/mask are split exactly
as in DCNv2: chunk(3) -> (o1, o2, mask), offset = cat(o1, o2), mask = sigmoid(mask).
"""
import math
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


class DCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, deformable_groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        n = in_channels * self.kernel_size[0] * self.kernel_size[1]
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        ch = deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(in_channels, ch, kernel_size=self.kernel_size,
                                          stride=self.stride, padding=self.padding, bias=True)
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(x, offset, self.weight, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation, mask=mask)
