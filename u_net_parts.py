# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 activation=nn.ELU):
        super(SeparableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.block1 = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, 1, bias=True),
                                    activation(),
                                    nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, self.stride,
                                              self.padding, self.dilation, self.out_channels, True),
                                    )

    def forward(self, input):
        return self.block1(input)


class SeparableConvTransposed2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 activation=nn.ELU):
        super(SeparableConvTransposed2d, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.block1 = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, 1, bias=True),
                                    activation(),
                                    nn.ConvTranspose2d(self.out_channels, self.out_channels, self.kernel_size,
                                                       self.stride,
                                                       self.padding, self.output_padding, self.out_channels, True),
                                    )

    def forward(self, input):
        return self.block1(input)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            SeparableConv2d(in_ch, out_ch, 3, padding=1),
            nn.ELU(alpha=0.1, inplace=True),
            nn.BatchNorm2d(out_ch),
            SeparableConv2d(out_ch, out_ch, 3, padding=1),
            nn.ELU(alpha=0.1, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


def calc_pad(kernel_size=3, dilation=1):
    kernel_size = (kernel_size, kernel_size) if type(kernel_size) == int else kernel_size
    dilation = (dilation, dilation) if type(dilation) == int else dilation
    return ((kernel_size[0] - 1) * dilation[0] / 2, (kernel_size[1] - 1) * dilation[1] / 2)


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        kernel_size = 3
        dilation = 2
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            SeparableConv2d(in_ch, in_ch, kernel_size=kernel_size, stride=2, padding=calc_pad(kernel_size, dilation),
                            dilation=dilation),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = SeparableConvTransposed2d(in_ch // 2, in_ch // 2, 3, stride=2, padding=1, output_padding=1)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
