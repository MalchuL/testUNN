import torch
import torch.nn as nn
import torch.nn.functional as func
from u_net_parts import *



class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, is_sigmoid=True):

        self.is_sigmoid = is_sigmoid
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

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
        if self.is_sigmoid:
            x = F.sigmoid(x)
        return x

if __name__ == "__main__":
    data = torch.empty(10, 3, 128, 128).uniform_(0, 1).cuda()
    segmentator = UNet(3,2).cuda()
    print(segmentator(data).shape)