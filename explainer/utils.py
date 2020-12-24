import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import subprocess as sp
import torch
from torch.nn import init

class ConditionalBatchNorm2d(nn.BatchNorm2d):
    """Conditional Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(ConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * output + bias


class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(CategoricalConditionalBatchNorm2d, self).forward(input, weight, bias)


class Downsampling(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.kernel_size = 2
        self.stride = 2

    def forward(self, x):
        #   input dimension (height or width)
        #   it is assumed that the input has the shape [batch_size, channels, height, width]
        dim_in = x.shape[2]
        #   padding = 'SAME'
        padding_h = int((dim_in * (self.stride - 1) + self.kernel_size - self.stride) / 2)
        padding_w = int((dim_in * (self.stride - 1) + self.kernel_size - self.stride) / 2)
        # x = F.pad(x, (padding_h * 2, padding_w * 2), 'constant', 0)
        # Чекай сайт https://pytorch.org/docs/stable/nn.functional.html, сначала паддим последнюю размерность с двух сторон, потом предыдущую. Или он вообще не нужен, хз
        # x = F.pad(x, [padding_h, padding_h, padding_w, padding_w], 'constant', 0)
        x = F.avg_pool2d(x, self.kernel_size, self.stride)

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
    def __init__(self, nums_class, n_channels):
        super().__init__()

        self.V = nn.Embedding(nums_class, n_channels)
        self.V = nn.utils.spectral_norm(self.V)

    def forward(self, x, y):
        temp = self.V(y)
        temp = torch.sum(temp * x, dim=1, keepdim=True)
        
        return temp


class GlobalSumPooling(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.sum(x, [2, 3])

        return x


class GeneratorEncoderResblock(pl.LightningModule):
    def get_gpu_memory(self):
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values[0]

    def __init__(self, in_channels, out_channels, num_classes, is_sn=False):
        super().__init__()

        self.identity = nn.Identity()
        self.bn1 = CategoricalConditionalBatchNorm2d(num_classes=num_classes, num_features=in_channels)
        self.relu = nn.ReLU()
        self.downsampling = Downsampling()
        self.conv1 = SpectralConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    is_sn=is_sn)
        self.bn2 = CategoricalConditionalBatchNorm2d(num_classes=num_classes, num_features=out_channels)
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
    def get_gpu_memory(self):
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values[0]

    def __init__(self, in_channels, out_channels, num_classes, is_sn=False):
        super().__init__()

        self.identity = nn.Identity()
        self.bn1 = CategoricalConditionalBatchNorm2d(num_classes=num_classes, num_features=in_channels)
        self.relu = nn.ReLU()
        self.upsampling = nn.Upsample(scale_factor=2)
        self.conv1 = SpectralConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    is_sn=is_sn)
        self.bn2 = CategoricalConditionalBatchNorm2d(num_classes=num_classes, num_features=out_channels)
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

        if is_sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        x = self.conv(x)

        return x
