import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch


class Upsampling(pl.LightningModule):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor, mode)

    def forward(self, x):
        x = self.upsampling(x)

        return x


class Downsampling(pl.LightningModule):
    def __init__(self, kernel_size, stride):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):

        #   input dimension (height or width)
        #   it is assumed that the input has the shape [batch_size, channels, height, width]
        dim_in = x.shape[2]

        #   padding = 'SAME'
        padding_h = int((dim_in * (self.stride[0] - 1) + self.kernel_size[0] - self.stride[0]) / 2)
        padding_w = int((dim_in * (self.stride[1] - 1) + self.kernel_size[1] - self.stride[1]) / 2)
        x = F.avg_pool2d(self.kernel_size, self.stride, padding=(padding_h, padding_w), input=x)

        return x


class Dense(pl.LightningModule):
    def __init__(self, in_channels, out_channels, is_sn=False):
        super().__init__()

        self.fc = nn.Linear(in_channels, out_channels)

        if is_sn:
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, x):
        x = self.fc(x)

        return x


class InnerProduct(pl.LightningModule):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.v = nn.init.xavier_uniform(torch.empty(size=[n_classes, in_channels]))

    def forward(self, x, y):
        self.v = self.v.transpose(0, 1)
        self.v = nn.utils.spectral_norm(self.v).transpose(0, 1)

        temp = torch.index_select(self.v, dim=0, index=y)
        temp = torch.sum(temp * x, 1, keepdim=True)

        return temp


class GlobalSumPooling(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.sum(x, [1, 2])

        return x


class GeneratorEncoderResblock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, is_sn=False):
        super().__init__()

        self.identity = nn.Identity()
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU()
        self.downsampling = Downsampling(kernel_size=(1, 2), stride=(1, 2))
        self.conv1 = SpectralConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, is_sn=is_sn)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = SpectralConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, is_sn=is_sn)
        self.conv_identity = SpectralConv2d(in_channels, out_channels, kernel_size=1, stride=1, is_sn=is_sn)

    def forward(self, x):
        temp = self.identity(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.downsampling(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        temp = self.downsampling(temp)
        temp = self.conv_identity(temp)

        x += temp

        return x


class GeneratorResblock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, is_sn=False):
        super().__init__()

        self.identity = nn.Identity()
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU()
        self.upsampling = Upsampling(scale_factor=2, mode='nearest')
        self.conv1 = SpectralConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    is_sn=is_sn)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = SpectralConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    is_sn=is_sn)
        self.conv_identity = SpectralConv2d(in_channels, out_channels, kernel_size=1, stride=1, is_sn=is_sn)

    def forward(self, x):
        temp = self.identity(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.upsampling(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        temp = self.upsampling(temp)
        temp = self.conv_identity(temp)

        x += temp

        return x


class DiscriminatorResBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, is_down=True, is_sn=True, is_first=False):
        super().__init__()

        self.identity = nn.Identity()
        self.is_down = is_down
        self.is_first = is_first
        self.relu = nn.ReLU()
        self.conv1 = SpectralConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, is_sn=is_sn)
        self.conv2 = SpectralConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, is_sn=is_sn)
        self.conv_identity = SpectralConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, is_sn=is_sn)
        self.downsampling = Downsampling(kernel_size=(1, 2), stride=(1, 2))

    def forward(self, x):
        temp = self.identity(x)

        if not self.is_first:
            x = self.relu(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        if self.is_down:
            x = self.downsampling(x)

            if self.is_first:
                temp = self.downsampling(temp)
                temp = self.conv3(temp)
            else:
                temp = self.conv3(temp)
                temp = self.downsampling(temp)

        return x + temp


class SpectralConv2d(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_sn=False):
        super().__init__()

        #   Assuming that stride = 1
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, padding, kernel_size, stride)

        if is_sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        x = self.conv(x)

        return x
