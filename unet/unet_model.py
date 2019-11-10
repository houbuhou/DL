from unet.unet_parts import *


basic_features = 64


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
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


class R_ResUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(R_ResUNet, self).__init__()
        self.inc = inconv(n_channels, basic_features)
        self.down1 = res_down(basic_features, 2 * basic_features)
        self.down2 = res_down(2 * basic_features, 4 * basic_features)
        self.down3 = res_down(4 * basic_features, 8 * basic_features)
        self.down4 = res_down(8 * basic_features, 8 * basic_features)

        self.up1 = up(16 * basic_features, 4 * basic_features)
        self.up2 = up(8 * basic_features, 2 * basic_features)
        self.up3 = up(4 * basic_features, basic_features)
        self.up4 = up(2 * basic_features, basic_features)
        self.aspp = ASPP(16 * basic_features)

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


if __name__ == '__main__':
    import torch
    import time
    TestTensor = torch.randn(1, 3, 512, 512).cuda()
    Net1 = DepthUNet().cuda()
    Net2 = UNet(n_channels=3, n_classes=1).cuda()
    start = time.time()
    output = Net2(TestTensor)
    print(output.shape)
    output2 = Net1(TestTensor)
    print(output2.shape)
    for i in range(100):
        output = Net2(TestTensor)
    end = time.time()
    print(end-start)
    start = time.time()
    for i in range(100):
        output2 = Net1(TestTensor)
    end = time.time()
    print(end-start)
