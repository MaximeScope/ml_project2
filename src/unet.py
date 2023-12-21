# from https://github.com/milesial/Pytorch-UNet/tree/master
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, cfg, bilinear=False):
        super(UNet, self).__init__()
        self.bilinear = bilinear
        self.depth = cfg.model.unet_depth
        self.has_sobel_filter = cfg.model.sobel_filter

        if self.has_sobel_filter:
            self.inc = DoubleConv(6, self.depth)
        else:
            self.inc = DoubleConv(3, self.depth)
        self.down1 = Down(self.depth, self.depth * 2)
        self.down2 = Down(self.depth * 2, self.depth * 2**2)
        self.down3 = Down(self.depth * 2**2, self.depth * 2**3)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.depth * 2**3, self.depth * 2**4 // factor)
        self.up1 = Up(self.depth * 2**4, self.depth * 2**3 // factor, bilinear)
        self.up2 = Up(self.depth * 2**3, self.depth * 2**2 // factor, bilinear)
        self.up3 = Up(self.depth * 2**2, self.depth * 2 // factor, bilinear)
        self.up4 = Up(self.depth * 2, self.depth, bilinear)
        self.outc = OutConv(self.depth, 1)

        # Define Sobel filter kernels
        self.sobel_x = (
            torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=cfg.device)
        ).repeat(3, 1, 1).unsqueeze_(1)
        self.sobel_y = (
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=cfg.device)
            .unsqueeze_(0)
        ).repeat(3, 1, 1).unsqueeze_(1)

    def forward(self, x):
        x = nn.LayerNorm(x.shape[1:], elementwise_affine=False)(x) 
        if self.has_sobel_filter:
            x = self.add_features(x)
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
        logits = torch.sigmoid(x)
        logits = torch.squeeze(logits, dim=1)
        return logits

    def add_features(self, x):
        # Add features to the input image
        # Sobel filter
        sobel_x = F.conv2d(x, self.sobel_x, groups=3, padding=1)
        sobel_y = F.conv2d(x, self.sobel_y, groups=3, padding=1)
        # Magnitude of the gradient
        sobel = torch.sqrt(sobel_x**2 + sobel_y**2)
        # Add the features to the input image
        x = torch.cat((x, sobel), dim=1)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
