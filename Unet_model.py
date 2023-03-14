import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("D:\\AppGallery\\python文件\\Unet(lxl)\\logs")


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # tensor  CHW
        H = torch.tensor([x2.size()[2]-x1.size()[2]])
        # 行差
        W = torch.tensor([x2.size()[3]-x1.size()[3]])
        # 列差
        x1 = torch.nn.functional.pad(x1, [H//2, H-H//2,
                                          W//2, W-W//2])
        x = torch.cat([x1, x2], dim=1)
        # 行维度跳跃连接
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.bilinear = True
        self.in_layer = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear=True)
        self.up2 = Up(512, 128, bilinear=True)
        self.up3 = Up(256, 64, bilinear=True)
        self.up4 = Up(128, 64, bilinear=True)
        self.out_layer = OutConv(64, 1)

    def forward(self, x):
        x1 = self.in_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        last_out = self.out_layer(x)
        return last_out


# if __name__ == '__main__':
net = Unet()
input = torch.ones(1, 1, 572, 572)
writer.add_graph(net, input)
writer.close()
output = net(input)
# print(output)
