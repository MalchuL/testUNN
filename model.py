import torch
import torch.nn as nn
import  torch.nn.functional as F

class MySegmentator(nn.Module):
    # zero padding claculation
    @staticmethod
    def calc_pad(kernel_size=3, dilation=1):
        kernel_size = (kernel_size, kernel_size) if type(kernel_size) == int else kernel_size
        dilation = (dilation, dilation) if type(dilation) == int else dilation
        return ((kernel_size[0] - 1) * dilation[0]/2, (kernel_size[1] - 1) * dilation[1]/2)

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



    def __init__(self):
        super(MySegmentator, self).__init__()
        #For Gaussian pyramid
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.default_kernel_size = default_kernel_size = 3
        self.default_padding = default_padding = self.calc_pad(self.default_kernel_size)
        # Conv2d is in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, default_kernel_size, padding=default_padding),
            nn.ReLU(),
            #squeze width by 2
            nn.Conv2d(64, 128, 3, dilation=2,stride=2, padding=self.calc_pad(3, 2), bias=False)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, default_kernel_size, padding=default_padding),
            nn.ReLU(),
            nn.Conv2d(128, 128, default_kernel_size, padding=default_padding),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, dilation=2, stride=2, padding=self.calc_pad(3, 2), bias=False)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 256, default_kernel_size, padding=default_padding),
            nn.ReLU(),
            nn.Conv2d(256, 256, default_kernel_size, padding=default_padding),
            nn.ReLU(),
            nn.Conv2d(256, 512, default_kernel_size, padding=default_padding)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(512, 512, default_kernel_size, padding=default_padding),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(512, 512, default_kernel_size, padding=default_padding),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(512, 512, default_kernel_size, padding=default_padding)
        )

        self.upscale2 = torch.nn.Upsample(scale_factor =2)
        self.upscale4 = torch.nn.Upsample(scale_factor=4)
    def forward(self, input):
        #Downsamples images and getting codes for it
        I0 = input
        F0 = self._encoder(I0)
        I1 = self._divide_size_by_2(I0)
        F1 = self.upscale2(self._encoder(I1))
        I2 = self._divide_size_by_2(I1)
        F2 = self.upscale4(self._encoder(I2))
        print(F0.shape,F1.shape,F2.shape)

if __name__ =="__main__":
    data = torch.empty(5,3,128,128).uniform_(0, 1)
    segmentator = MySegmentator()
    print(segmentator._divide_size_by_2(data).shape)
    print(segmentator(data).shape)