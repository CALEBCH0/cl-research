from torch.nn import (
    Conv2d,
    BatchNorm1d,
    BatchNorm2d,
    Sequential,
    Module,
)
import torch
import torch.nn as nn

from .common import ECA_Layer, SEBlock, CbamBlock, GCT
from .activation import get_activation_layer


# #################################  Original Arcface Model #############################################################


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# #################################  MobileFaceNet #############################################################


class Conv_block(Module):
    def __init__(
        self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, activation='relu'
    ):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)
        self.acti = get_activation_layer(activation, out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.acti(x)
        return x


class Linear_block(Module):
    def __init__(
        self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1
    ):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
    def __init__(
        self,
        in_c,
        out_c,
        attention,
        residual=False,
        kernel=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        groups=1,
    ):
        super(Depth_Wise, self).__init__()

        self.conv = Conv_block(in_c, out_c=groups, kernel=(
            1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(
            groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(
            1, 1), padding=(0, 0), stride=(1, 1))

        self.attention = attention  # se, eca, cbam
        if self.attention == "eca":
            # Efficient Channel Attention https://eehoeskrap.tistory.com/480
            self.attention_layer = ECA_Layer(out_c)
        elif self.attention == "se":
            self.attention_layer = SEBlock(out_c)
        elif self.attention == "cbam":
            self.attention_layer = CbamBlock(out_c)
        elif self.attention == "gct":
            # Gated Channel Transformation https://paperswithcode.com/method/gct
            self.attention_layer = GCT(out_c)

        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.attention != "none":
            x = self.attention_layer(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(
        self,
        c,
        attention,
        num_block,
        groups,
        kernel=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
    ):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Depth_Wise(
                    c,
                    c,
                    attention,
                    residual=True,
                    kernel=kernel,
                    padding=padding,
                    stride=stride,
                    groups=groups,
                )
            )
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class GNAP(Module):  # Global Norm-Aware Pooling
    def __init__(self, embedding_size):
        super(GNAP, self).__init__()
        assert embedding_size == 512
        self.bn1 = BatchNorm2d(512)  # , affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.bn2 = BatchNorm1d(512, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature


class GDC(nn.Module):
    def __init__(self, in_c, embedding_size):
        super(GDC, self).__init__()  # 512               # 16,16
        self.conv_6_dw = Linear_block(in_c, in_c, kernel=(16, 16),
                                      stride=(1, 1), padding=(0, 0), groups=in_c)
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(in_c, embedding_size, bias=False)
        # self.bn = BatchNorm1d(embedding_size, affine=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


class Modified_MobileFaceNet(Module):
    def __init__(
        self, input_size=(1, 256, 256), num_features=512, output_name="GDC", attention="none", activation='relu'
    ):
        super(Modified_MobileFaceNet, self).__init__()
        # assert output_name in ["GNAP", "GDC"]
        # assert input_size[0] in [112]
        # 1
        self.conv1 = Conv_block(1, 64, kernel=(
            3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(
            64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, activation=activation
        )
        self.conv_23 = Depth_Wise(
            64, 64, attention, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128
        )
        self.conv_3 = Residual(
            64,
            attention,
            num_block=4,
            groups=128,
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.conv_34 = Depth_Wise(
            64, 128, attention, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256
        )
        self.conv_4 = Residual(
            128,
            attention,
            num_block=6,
            groups=256,
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.conv_45 = Depth_Wise(
            128,
            128,
            attention,
            kernel=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            groups=512,
        )
        self.conv_5 = Residual(
            128,
            attention,
            num_block=2,
            groups=256,
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.conv_6_sep = Conv_block(
            128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0)
        )

        self.output_name = output_name
        if output_name == "GNAP":
            self.output_layer = GNAP(512)
        elif output_name == "GDC":
            self.output_layer = GDC(512, num_features)
        elif output_name == "v3":
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Conv2d(512, num_features, 1, 1, 0),
                # nn.Linear(576, last_channel),
                # nn.Linear(exp_channel, output_channel),
                get_activation_layer(activation, num_features),
            )
        else:
            raise NotImplementedError

        self._initialize_weights()
        # self.metric_fc = metric_fc

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # _input = x[:,:,:,:-1]
        # _label = x[:,0,0,-1]

        out = self.conv1(x)  # Conv_block

        out = self.conv2_dw(out)  # Conv_block

        out = self.conv_23(out)  # Depth_Wise

        out = self.conv_3(out)  # Residual

        out = self.conv_34(out)  # Depth_Wise

        out = self.conv_4(out)  # Residual

        out = self.conv_45(out)  # Depth_Wise

        out = self.conv_5(out)  # Residual

        conv_features = self.conv_6_sep(out)  # Conv_block

        if self.output_name == "v3":
            avgpool = self.avgpool(conv_features)
            out = self.classifier(avgpool)
        else:
            out = self.output_layer(conv_features)  # GDC/GNAP/v3

        # out = self.metric_fc(out, _label)

        return out


def modified_mbilefacenet(**kwargs):
    return Modified_MobileFaceNet(**kwargs)
