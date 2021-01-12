import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F


class ConditionalBatchNorm2d(pl.LightningModule):
    def __init__(self, nums_class, num_features):
        super().__init__()

        self.num_features = num_features

        self.bn = nn.BatchNorm2d(num_features=num_features, momentum=0.5, eps=1e-3)

        self.beta = nn.Parameter(torch.zeros(size=[num_features]))
        self.gamma = nn.Parameter(torch.ones(size=[num_features]))

        self.beta_conditional = nn.Embedding(nums_class, num_features)
        nn.init.zeros_(self.beta_conditional.weight.data)

        self.gamma_conditional = nn.Embedding(nums_class, num_features)
        nn.init.ones_(self.gamma_conditional.weight.data)

    def forward(self, x, y=None):
        if y is None:
            x = self.bn(x)
        else:
            beta, gamma = self.beta_conditional(y.long()), self.gamma_conditional(y.long())
            beta = torch.reshape(beta, [-1, self.num_features, 1, 1])
            gamma = torch.reshape(gamma, [-1, self.num_features, 1, 1])

            x = self.bn(x)
            x = gamma * x + beta

        return x


class Downsampling(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.kernel_size = 2
        self.stride = 2

    def forward(self, x):
        x = F.avg_pool2d(x, self.kernel_size, self.stride)

        return x


class Dense(pl.LightningModule):
    def __init__(self, in_channels, out_channels, is_sn=False):
        super().__init__()

        self.fc = nn.Linear(in_channels, out_channels)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

        if is_sn:
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, x):
        x = self.fc(x)

        return x


class InnerProduct(pl.LightningModule):
    def __init__(self, nums_class, n_channels):
        super().__init__()

        self.V = nn.Embedding(nums_class, n_channels)
        nn.init.xavier_uniform_(self.V.weight.data)
        self.V = nn.utils.spectral_norm(self.V)

    def forward(self, x, y):
        temp = self.V(y)
        temp = torch.sum(temp * x, dim=1, keepdim=True)
        return temp


class GlobalSumPooling(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.sum(x, [2, 3], keepdim=False)

        return x


class GeneratorEncoderResblock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, num_classes, is_sn=False):
        super().__init__()

        self.identity = nn.Identity()
        self.bn1 = ConditionalBatchNorm2d(nums_class=num_classes, num_features=in_channels)
        self.relu = nn.ReLU()
        self.downsampling = Downsampling()
        self.conv1 = SpectralConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    is_sn=is_sn)
        self.bn2 = ConditionalBatchNorm2d(nums_class=num_classes, num_features=out_channels)
        self.conv2 = SpectralConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    is_sn=is_sn)
        self.conv_identity = SpectralConv2d(in_channels, out_channels, kernel_size=1, stride=1, is_sn=is_sn)

    def forward(self, x, y):
        temp = self.identity(x)
        x = self.bn1(x, y)
        x = self.relu(x)
        x = self.downsampling(x)
        x = self.conv1(x)
        x = self.bn2(x, y)
        x = self.relu(x)
        x = self.conv2(x)

        temp = self.downsampling(temp)
        temp = self.conv_identity(temp)
        x += temp

        return x


class GeneratorResblock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, num_classes, is_sn=False):
        super().__init__()

        self.identity = nn.Identity()
        self.bn1 = ConditionalBatchNorm2d(nums_class=num_classes, num_features=in_channels)
        self.relu = nn.ReLU()
        self.upsampling = nn.Upsample(scale_factor=2)
        self.conv1 = SpectralConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    is_sn=is_sn)
        self.bn2 = ConditionalBatchNorm2d(nums_class=num_classes, num_features=out_channels)
        self.conv2 = SpectralConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    is_sn=is_sn)
        self.conv_identity = SpectralConv2d(in_channels, out_channels, kernel_size=1, stride=1, is_sn=is_sn)

    def forward(self, x, y):
        temp = self.identity(x)

        x = self.bn1(x, y)
        x = self.relu(x)
        x = self.upsampling(x)
        x = self.conv1(x)
        x = self.bn2(x, y)
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
        self.conv1 = SpectralConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    is_sn=is_sn)
        self.conv2 = SpectralConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    is_sn=is_sn)
        self.conv_identity = SpectralConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                            is_sn=is_sn)
        self.downsampling = Downsampling()

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
                temp = self.conv_identity(temp)
            else:
                temp = self.conv_identity(temp)
                temp = self.downsampling(temp)

        return x + temp


class SpectralConv2d(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_sn=False):
        super().__init__()

        #   Assuming that stride = 1
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.0)

        if is_sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        x = self.conv(x)

        return x
