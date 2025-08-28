import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import make_divisible, h_sigmoid, ModifiedGDC

"""
https://github.com/Hazqeel09/ellzaf_ml/blob/main/ellzaf_ml/models/ghostfacenetsv2.py
"""


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.PReLU, gate_fn=h_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn(inplace=False)
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.PReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModuleV2(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, prelu=True, mode=None):
        super(GhostModuleV2, self).__init__()
        self.mode = mode
        self.gate_fn = nn.Sigmoid()

        if self.mode in ['original']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.PReLU() if prelu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.PReLU() if prelu else nn.Sequential(),
            )
        elif self.mode in ['attn']:  # DFC
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.PReLU() if prelu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.PReLU() if prelu else nn.Sequential(),
            )
            self.short_conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :]
        elif self.mode in ['attn']:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :] * F.interpolate(self.gate_fn(res), size=(out.shape[-2], out.shape[-1]), mode='nearest')


class GhostBottleneckV2(nn.Module):
    def __init__(self, in_chs,
                 mid_chs,
                 out_chs,
                 dw_kernel_size=3,
                 stride=1,
                 se_ratio=0.,
                 layer_id=None,
                 ):
        super(GhostBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        if layer_id <= 1:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, prelu=True, mode='original')
        else:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, prelu=True, mode='attn')

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride, padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        self.ghost2 = GhostModuleV2(mid_chs, out_chs, prelu=False, mode='original')

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride, padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x


class GhostFaceNetV2(nn.Module):
    def __init__(self,
                 cfgs=None,
                 image_size=256,
                 num_features=0,
                 width=1.0,
                 channels=1,  # IR
                 dropout=0.2,
                 block=GhostBottleneckV2,
                 add_pointwise_conv=False,
                 bn_momentum=0.9,
                 bn_epsilon=1e-5,
                 init_kaiming=True,
                 ):
        super(GhostFaceNetV2, self).__init__()

        if cfgs is None:
            self.cfgs = [
                # k, t, c, SE, s
                [[3, 16, 16, 0, 1]],
                [[3, 48, 24, 0, 2]],
                [[3, 72, 24, 0, 1]],
                [[5, 72, 40, 0.25, 2]],
                [[5, 120, 40, 0.25, 1]],
                [[3, 240, 80, 0, 2]],
                [[3, 200, 80, 0, 1],
                 [3, 184, 80, 0, 1],
                 [3, 184, 80, 0, 1],
                 [3, 480, 112, 0.25, 1],
                 [3, 672, 112, 0.25, 1]],
                [[5, 672, 160, 0.25, 2]],
                [[5, 960, 160, 0, 1],
                 [5, 960, 160, 0.25, 1],
                 [5, 960, 160, 0, 1],
                 [5, 960, 160, 0.25, 1]]
            ]
        else:
            self.cfgs = cfgs

        # building first layer
        output_channel = make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(channels, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.PReLU()
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        layer_id = 0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = make_divisible(c * width, 4)
                hidden_channel = make_divisible(exp_size * width, 4)
                if block == GhostBottleneckV2:
                    layers.append(block(input_channel, hidden_channel, output_channel, k, s, se_ratio=se_ratio, layer_id=layer_id))
                input_channel = output_channel
                layer_id += 1
            stages.append(nn.Sequential(*layers))

        output_channel = make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        pointwise_conv = []
        if add_pointwise_conv:
            pointwise_conv.append(nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True))
            pointwise_conv.append(nn.BatchNorm2d(output_channel))
            pointwise_conv.append(nn.PReLU())
        else:
            pointwise_conv.append(nn.Sequential())

        self.pointwise_conv = nn.Sequential(*pointwise_conv)
        self.output_layer = ModifiedGDC(image_size, output_channel, num_features, dropout)

        # Initialize weights
        for m in self.modules():
            if init_kaiming:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    negative_slope = 0.25  # Default value for PReLU in PyTorch, change it if you use custom value
                    m.weight.data.normal_(0, math.sqrt(2. / (fan_in * (1 + negative_slope ** 2))))
            if isinstance(m, nn.BatchNorm2d):
                m.momentum, m.eps = bn_momentum, bn_epsilon

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.pointwise_conv(x)
        x = self.output_layer(x)
        return x


def ghostfacenetv2(image_size, num_features, width):
    return GhostFaceNetV2(image_size=image_size, num_features=num_features, width=width)
