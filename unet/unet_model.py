from .unet_parts import *


basic_features = 64


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, basic_features)

        self.down1 = down(basic_features, 2 * basic_features)
        self.down2 = down(2 * basic_features, 4 * basic_features)
        self.down3 = down(4 * basic_features, 8 * basic_features)
        self.down4 = down(8 * basic_features, 8 * basic_features)

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