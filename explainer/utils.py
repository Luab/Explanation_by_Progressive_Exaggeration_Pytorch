import pytorch_lightning as pl
import torch.nn as nn
import torch


class Dense(pl.LightningModule):
    def __init__(self, in_channels, out_channels, is_sn=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_sn = is_sn

    def forward(self, x):
        w = nn.init.xavier_uniform(torch.empty(size=[self.in_channels, self.out_channels]))

        if self.is_sn:
            w = nn.utils.spectral_norm(w)

        return torch.matmul(x, w)


class InnerProduct(pl.LightningModule):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels

    def forward(self, x, y, n_classes):
        v = nn.init.xavier_uniform(torch.empty(size=[n_classes, self.in_channels])).transpose(0, 1)
        v = nn.utils.spectral_norm(v).transpose(0, 1)

        temp = torch.index_select(input=v, dim=0, index=y)
        temp = torch.sum(temp * x, 1, keepdim=True)

        return temp


class GlobalSumPooling(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.sum(x, [1, 2])

        return x


class DiscriminatorResBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, is_down=True, is_sn=True, is_first=False):
        super().__init__()

        self.out_channels = out_channels
        self.is_down = is_down
        self.is_first = is_first
        self.relu = nn.ReLU()
        self.conv1 = SpectralConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, is_sn=is_sn)
        self.conv2 = SpectralConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, is_sn=is_sn)
        self.conv3 = SpectralConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, is_sn=is_sn)

    def forward(self, x):
        temp = nn.Identity(x)

        if not self.is_first:
            x = self.relu(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        #   In the original implementation different kernel sizes and strides are applied for different dimensions
        #   TODO check the kernel sizes, strides, and padding
        if self.is_down:
            x = nn.AvgPool2d(kernel_size=2, padding=0)(x)

            if self.is_first:
                temp = nn.AvgPool2d(kernel_size=2, padding=0)(temp)
                temp = self.conv3(temp)
            else:
                temp = self.conv3(temp)
                temp = nn.AvgPool2d(kernel_size=2, padding=0)(temp)

        return x + temp


class SpectralConv2d(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_sn=False):
        super().__init__()

        #   Assuming that stride = 1
        padding = int(kernel_size - 1) / 2
        self.conv = nn.Conv2d(in_channels, out_channels, padding, kernel_size, stride)
        self.is_sn = is_sn

    def forward(self, x):
        x = self.conv(x)

        if self.is_sn:
            x = nn.utils.spectral_norm(x)

        return x
