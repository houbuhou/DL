# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
# from .aspp import build_aspp


class DepthwiseSeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, groups=1, bias=False):
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


class DepthDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthDoubleConv, self).__init__()
        self.DepthConv = nn.Sequential(
            DepthwiseSeparableConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.outconv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
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
