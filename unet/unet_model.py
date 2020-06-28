from unet.unet_parts import *


basic_features = 64


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class InvertedResidualUp(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidualUp, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_transconv(inp, inp, kernel_size=self.stride, stride=self.stride, padding=0),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_transconv(branch_features, branch_features, kernel_size=self.stride, stride=self.stride, padding=0),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

        self.outconv = nn.Conv2d(2 * oup, oup, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def depthwise_transconv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.ConvTranspose2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x, x_external):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x), x_external), dim=1)

        out = channel_shuffle(out, 4)

        out = self.outconv(out)

        return out


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, basic_features)

        self.down1 = down(basic_features, 2 * basic_features)
        self.down2 = down(2 * basic_features, 4 * basic_features)
        self.down3 = down(4 * basic_features, 8 * basic_features)
        self.down4 = down(8 * basic_features, 16 * basic_features)

        self.up1 = up(16 * basic_features, 8 * basic_features)
        self.up2 = up(8 * basic_features, 4 * basic_features)
        self.up3 = up(4 * basic_features, 2 * basic_features)
        self.up4 = up(2 * basic_features, 1 * basic_features)

        self.outc = outconv(basic_features, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x)
        return x


class DepthUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(DepthUNet, self).__init__()
        self.InConv = DepthInConv(n_channels, basic_features)
        self.Down1 = DepthDown(1 * basic_features, 2 * basic_features)
        self.Down2 = DepthDown(2 * basic_features, 4 * basic_features)
        self.Down3 = DepthDown(4 * basic_features, 8 * basic_features)
        self.Down4 = DepthDown(8 * basic_features, 16 * basic_features)

        self.Up1 = DepthUp(16 * basic_features, 8 * basic_features)
        self.Up2 = DepthUp(8 * basic_features, 4 * basic_features)
        self.Up3 = DepthUp(4 * basic_features, 2 * basic_features)
        self.Up4 = DepthUp(2 * basic_features, 1 * basic_features)

        self.OutConv = DepthOutConv(basic_features, n_classes)

    def forward(self, x):
        x1 = self.InConv(x)
        x2 = self.Down1(x1)
        x3 = self.Down2(x2)
        x4 = self.Down3(x3)
        x5 = self.Down4(x4)

        x6 = self.Up1(x5, x4)
        x7 = self.Up2(x6, x3)
        x8 = self.Up3(x7, x2)
        x9 = self.Up4(x8, x1)

        output = self.OutConv(x9)

        return output


class ShuffleUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(ShuffleUNet, self).__init__()
        self.inconv = nn.Conv2d(n_channels, basic_features, kernel_size=3, stride=1, padding=1)
        self.down1 = nn.Sequential(
            InvertedResidual(basic_features, basic_features, 1),
            InvertedResidual(basic_features, 2 * basic_features, 2)
        )
        self.down2 = nn.Sequential(
            InvertedResidual(2 * basic_features, 2 * basic_features, 1),
            InvertedResidual(2 * basic_features, 4 * basic_features, 2)
        )
        self.down3 = nn.Sequential(
            InvertedResidual(4 * basic_features, 4 * basic_features, 1),
            InvertedResidual(4 * basic_features, 8 * basic_features, 2)
        )
        self.down4 = nn.Sequential(
            InvertedResidual(8 * basic_features, 8 * basic_features, 1),
            InvertedResidual(8 * basic_features, 16 * basic_features, 2)
        )
        self.mid = InvertedResidual(16 * basic_features, 16 * basic_features, 1)

        self.up4 = InvertedResidualUp(16 * basic_features, 8 * basic_features, 2)
        self.upconv4 = InvertedResidual(8 * basic_features, 8 * basic_features, 1)

        self.up3 = InvertedResidualUp(8 * basic_features, 4 * basic_features, 2)
        self.upconv3 = InvertedResidual(4 * basic_features, 4 * basic_features, 1)

        self.up2 = InvertedResidualUp(4 * basic_features, 2 * basic_features, 2)
        self.upconv2 = InvertedResidual(2 * basic_features, 2 * basic_features, 1)

        self.up1 = InvertedResidualUp(2 * basic_features, 1 * basic_features, 2)
        self.upconv1 = InvertedResidual(1 * basic_features, 1 * basic_features, 1)

        self.outconv = nn.Conv2d(basic_features, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.inconv(x)     # 3 -> 64
        x2 = self.down1(x1)     # 64 -> 128
        x3 = self.down2(x2)     # 128 -> 256
        x4 = self.down3(x3)     # 256 -> 512
        x5 = self.down4(x4)     # 512 -> 1024

        x6 = self.mid(x5)     # 1024 -> 1024

        x7_1 = self.up4(x6, x4)     # 1024 -> 512
        x7 = self.upconv4(x7_1)
        x8_1 = self.up3(x7, x3)     # 512 -> 256
        x8 = self.upconv3(x8_1)
        x9_1 = self.up2(x8, x2)     # 256 -> 128
        x9 = self.upconv2(x9_1)
        x10_1 = self.up1(x9, x1)     # 128 -> 64
        x10 = self.upconv1(x10_1)

        x11 = self.outconv(x10)

        return x11


if __name__ == '__main__':
    import torch
    import time
    TestTensor = torch.randn(1, 3, 512, 512).cuda()
    TestTensor1 = torch.randn(1, 2, 512, 512).cuda()
    Net1 = UNet().cuda()
    Net2 = ShuffleUNet().cuda()
    # Net2 = UNet(n_channels=4, n_classes=1).cuda()
    start = time.time()
    output = Net2(TestTensor)
    output2 = Net1(TestTensor)
    del output
    del output2
    for i in range(200):
        output = Net1(TestTensor)
        del output
    end = time.time()
    print(end-start)
    start = time.time()
    for i in range(200):
        output2 = Net2(TestTensor)
        del output2
    end = time.time()
    print(end-start)






