import torch
import torch.nn as nn
import torch.nn.functional as func


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, activation = nn.Tanhshrink):
        super(SeparableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.block1 = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False),
                                    #activation(),
                                    nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, self.stride,
                                              self.padding, self.dilation, self.out_channels, True),
                                    )

    def forward(self, input):
        return self.block1(input)



class SeparableConvTransposed2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, activation = nn.Tanhshrink):
        super(SeparableConvTransposed2d, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.block1 = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False),
                                    #activation(),
                                    nn.ConvTranspose2d(self.out_channels, self.out_channels, self.kernel_size, self.stride,
                                              self.padding, self.output_padding, self.out_channels, True),
                                    )

    def forward(self, input):
        return self.block1(input)


class MySegmentator(nn.Module):
    # zero padding claculation
    @staticmethod
    def calc_pad(kernel_size=3, dilation=1):
        kernel_size = (kernel_size, kernel_size) if type(kernel_size) == int else kernel_size
        dilation = (dilation, dilation) if type(dilation) == int else dilation
        return ((kernel_size[0] - 1) * dilation[0] / 2, (kernel_size[1] - 1) * dilation[1] / 2)

    def _set_start_params(self):
        pass

    def _divide_size_by_2(self, input):
        return self.avg_pool(input)

    def _encoder(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    def _decoder(self, input):
        x = self.block5(input)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        return x

    def __init__(self, sigmoid_output=True):
        super(MySegmentator, self).__init__()

        self.sigmoid_output = sigmoid_output

        # For Gaussian pyramid
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.default_kernel_size = default_kernel_size = 3
        self.default_padding = default_padding = self.calc_pad(self.default_kernel_size)
        # Conv2d is in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        self.block1 = nn.Sequential(
            SeparableConv2d(3, 64, default_kernel_size, padding=default_padding),
            nn.ELU(alpha=0.1),
            # squeze width by 2
            SeparableConv2d(64, 128, 3, dilation=2, stride=2, padding=self.calc_pad(3, 2)),
            nn.ELU(alpha=0.1)
        )
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(128),
            SeparableConv2d(128, 128, default_kernel_size, padding=default_padding),
            nn.ELU(alpha=0.1),
            SeparableConv2d(128, 256, 3, dilation=2, stride=2, padding=self.calc_pad(3, 2)),
            nn.ELU(alpha=0.1)
        )
        self.block3 = nn.Sequential(
            nn.BatchNorm2d(256),
            SeparableConv2d(256, 256, default_kernel_size, padding=default_padding),
            nn.ELU(alpha=0.1),
            SeparableConv2d(256, 256, default_kernel_size, padding=default_padding),
            nn.ELU(alpha=0.1),
            SeparableConv2d(256, 256, default_kernel_size, padding=default_padding),
            nn.ELU(alpha=0.1)
        )
        self.block4 = nn.Sequential(
            nn.BatchNorm2d(256),
            SeparableConv2d(256, 512, default_kernel_size, padding=default_padding),
            nn.ELU(alpha=0.1),
            nn.Dropout2d(),
            SeparableConv2d(512, 512, default_kernel_size, padding=default_padding),
            nn.ELU(alpha=0.1),
            nn.Dropout2d(),
            SeparableConv2d(512, 512, default_kernel_size, padding=default_padding),
            nn.ELU(alpha=0.1)
        )

        # between operators
        self.upscale2 = torch.nn.Upsample(scale_factor=2)
        self.upscale4 = torch.nn.Upsample(scale_factor=4)
        self.transfer_to_decoder1 = nn.ConvTranspose2d(512 * 3, 64, 1)
        #self.transfer_to_decoder2 = nn.ConvTranspose2d(512, 256, 1)

        # decoder
        self.block5 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 1),
            nn.ELU(alpha=0.1),
            SeparableConvTransposed2d(64, 64, default_kernel_size, padding=default_padding),
            nn.ELU(alpha=0.1),
            nn.ConvTranspose2d(64, 512, 1),
            nn.ELU(alpha=0.1)
        )
        self.block6 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 64, 1),
            nn.ELU(alpha=0.1),
            SeparableConvTransposed2d(64, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ELU(alpha=0.1),
            nn.ConvTranspose2d(64, 256, 1),
            nn.ELU(alpha=0.1)
        )
        self.block7 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 64, 1),
            nn.ELU(alpha=0.1),
            SeparableConvTransposed2d(64, 64, default_kernel_size, padding=default_padding),
            nn.ELU(alpha=0.1),
            nn.ConvTranspose2d(64, 128, 1),
            nn.ELU(alpha=0.1)
        )
        self.block8 = nn.Sequential(
            nn.BatchNorm2d(128),
            SeparableConvTransposed2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ELU(alpha=0.1)
        )
        self.block9 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 1),
   #         nn.Sigmoid()
        )

    def forward(self, input):
        # Downsamples images and encodes for it
        I0 = input
        F0 = self._encoder(I0)
        I1 = self._divide_size_by_2(I0)
        F1 = self.upscale2(self._encoder(I1))
        I2 = self._divide_size_by_2(I1)
        F2 = self.upscale4(self._encoder(I2))
        F = torch.cat([F0, F1, F2], dim=1)
        F = self.transfer_to_decoder1(F)
        F = func.elu(F)
        #F = self.transfer_to_decoder2(F)
        #F = func.elu(F)
        decoded = self._decoder(F)
        if self.sigmoid_output:
            decoded = func.sigmoid(decoded)
        return decoded


if __name__ == "__main__":
    data = torch.empty(10, 3, 32, 32).uniform_(0, 1).cuda()
    segmentator = MySegmentator().cuda()
    print(segmentator(data).max())
