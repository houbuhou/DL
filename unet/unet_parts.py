# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F



class DepthwiseSeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, groups=1, bias=True):
        super(DepthwiseSeparableConv2D, self).__init__()

        self.DepthwiseConv = nn.Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=kernel_size,
                                       padding=padding,
                                       stride=stride,
                                       dilation=dilation,
                                       groups=in_channels,
                                       bias=bias)

        self.PointwiseConv = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=1,
                                       padding=0,
                                       stride=1,
                                       dilation=1,
                                       groups=1,
                                       bias=bias)

    def forward(self, x):
        x = self.DepthwiseConv(x)
        x = self.PointwiseConv(x)
        return x


class NonLocalFeatureFusionPart(nn.Module):
    def __init__(self, low_in_channels, high_in_channels, out_channels, key_channels=1, value_channels=1):
        super(NonLocalFeatureFusionPart, self).__init__()
        self.low_in_channels = low_in_channels
        self.high_in_channels = high_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.ConvKey = nn.Conv2d(in_channels=self.low_in_channels,
                                 out_channels=self.key_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.ConvQuery = nn.Conv2d(in_channels=self.high_in_channels,
                                   out_channels=self.key_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        self.ConvValue = nn.Conv2d(in_channels=self.low_in_channels,
                                   out_channels=self.value_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

        self.W = nn.Conv2d(in_channels=self.value_channels,
                           out_channels=self.out_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        nn.init.constant_(self.W.bias, 0)
        nn.init.constant_(self.W.weight, 0)

    def forward(self, x_low, x_high):
        BatchSize = x_low.size(0)
        query = self.ConvQuery(x_high).view(BatchSize, self.key_channels, -1)
        key = self.ConvKey(x_low).view(BatchSize, self.key_channels, -1)
        value = self.ConvValue(x_low).view(BatchSize, self.value_channels, -1)
        query = query.permute(0, 2, 1)
        value = value.permute(0, 2, 1)

        SimMap = torch.matmul(query, key)
        # SimMap = SimMap * (self.key_channels ** -0.5)
        SimMap = F.softmax(SimMap, dim=-1)

        context = torch.matmul(SimMap, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(BatchSize, self.value_channels, *x_high.size()[2:])
        context = self.W(context)

        return context + x_high


class ASPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, basic_channels=256):
        super(ASPPBottleneck, self).__init__()
        self.Conv1x1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                               out_channels=basic_channels,
                                               kernel_size=1,
                                               stride=1,
                                               padding=0,
                                               dilation=1),
                                     nn.BatchNorm2d(basic_channels),
                                     nn.ReLU())

        self.Conv3x3D6 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=basic_channels,
                      kernel_size=3,
                      stride=1,
                      padding=6,
                      dilation=6),
            nn.BatchNorm2d(basic_channels),
            nn.SELU()
        )
        self.Conv3x3D12 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=basic_channels,
                      kernel_size=3,
                      stride=1,
                      padding=12,
                      dilation=12),
            nn.BatchNorm2d(basic_channels),
            nn.SELU()
        )
        self.Conv3x3D18 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=basic_channels,
                      kernel_size=3,
                      stride=1,
                      padding=18,
                      dilation=18),
            nn.BatchNorm2d(basic_channels),
            nn.SELU()
        )

        self.AveragePool = nn.AdaptiveAvgPool2d(2)          # My problem is that why the output dimension is 1
        self.Conv1x1Avg = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=basic_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      dilation=1),
            nn.BatchNorm2d(basic_channels),
            nn.SELU()
        )

        self.OutConv = nn.Sequential(
            nn.Conv2d(
                in_channels=5*basic_channels,
                out_channels=basic_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1
            ),
            nn.BatchNorm2d(basic_channels),
            nn.SELU(),
            nn.Conv2d(
                in_channels=basic_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.SELU()
        )

    def forward(self, x):
        BatchSize, C, H, W = x.size()
        x1 = self.Conv1x1(x)
        x2 = self.Conv3x3D6(x)
        x3 = self.Conv3x3D12(x)
        x4 = self.Conv3x3D18(x)

        x_avg = self.AveragePool(x)
        x5 = self.Conv1x1Avg(x_avg)
        x5 = F.upsample(x5, size=(H, W), mode="bilinear")

        output = torch.cat([x1, x2, x3, x4, x5], dim=1)
        output = self.OutConv(output)

        return output


class DepthDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthDoubleConv, self).__init__()
        self.DepthConv = nn.Sequential(
            DepthwiseSeparableConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            DepthwiseSeparableConv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        output = self.DepthConv(x)

        return output


class DepthInConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthInConv, self).__init__()
        self.DepthConv = nn.Sequential(
            DepthwiseSeparableConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            DepthwiseSeparableConv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            DepthwiseSeparableConv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        output = self.DepthConv(x)

        return output


class DepthDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthDown, self).__init__()
        self.DepthDownUnit = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DepthDoubleConv(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        output = self.DepthDownUnit(x)
        return output


class DepthUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthUp, self).__init__()
        self.DepthUpUnit = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2,
                                              stride=2)
        self.DepthConv = DepthDoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x_expand, x_copy):
        x_up = self.DepthUpUnit(x_expand)
        x_concat = torch.cat([x_up, x_copy], dim=1)
        output = self.DepthConv(x_concat)

        return output


class DepthOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthOutConv, self).__init__()
        self.DepthConv = DepthwiseSeparableConv2D(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=1)
        self.Sigmoid = nn.Sigmoid()
        self.BN = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.DepthConv(x)
        x2 = self.BN(x1)
        output = self.Sigmoid(x2)

        return output


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, dilation=1):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.outconv = nn.Conv2d(in_ch, out_ch, 3, padding=1, dilation=dilation)
        self.BN = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        return out


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=1),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)
        self.double_conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        # x = self.conv(x)
        # x = self.relu(x)
        x = self.double_conv(x)
        return x


# class ASPP(nn.Module):
#     def __init__(self, in_ch):
#         super(ASPP, self).__init__()
#         self.aspp = build_aspp(backbone='resnet', output_stride=8, BatchNorm=nn.BatchNorm2d)


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class res_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(res_down, self).__init__()
        self.mpconv = nn.Sequential(
            double_conv(in_ch, in_ch)
        )
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        res = x
        x = self.mpconv(x)
        x = self.maxpool(x + res)
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        # if bilinear:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # else:
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print('x1',x1.size())
        # diffX = x1.size()[2] - x2.size()[2]
        # diffY = x1.size()[3] - x2.size()[3]
        # x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
        #                 diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        # print(x.size())
        x = self.conv(x)
        return x



class res_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(res_up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, in_ch)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print('x1',x1.size())
        # diffX = x1.size()[2] - x2.size()[2]
        # diffY = x1.size()[3] - x2.size()[3]
        # x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
        #                 diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        # print(x.size())
        res = x
        x = self.conv(x)
        x = x + res
        x = self.conv1(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        out_ch1 = int(in_ch / 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch1, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
        )
        self.Sigmoid = nn.Sigmoid()
        # self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.Sigmoid(x)
        return x


