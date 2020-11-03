import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import subprocess as sp
import torch


class ConditionalBatchNorm2d(nn.Module):

    def __init__(self, input_shape: int, num_classes: int):
        """
        Creates 2 type of variables for y is None and y is not None
        :param input_shape: it is x.shape[-1], we need to know it in at compile-time, I think we could guess it in debugging
        :param num_classes: NUmber of outputting features
        """
        super().__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.beta1 = torch.zeros(size=[input_shape], requires_grad=True)
        self.gamma1 = torch.ones(size=[input_shape], requires_grad=True)

        self.beta2 = torch.zeros(size=[num_classes, input_shape], requires_grad=True)
        self.gamma2 = torch.ones(size=[num_classes, input_shape], requires_grad=True)

    def forward(self, x, y=None):
        if y is not None:
            # self.beta1 = torch.reshape(torch.index_select(self.beta2, 0, y), [-1, 1, 1, self.input_shape])
            # self.gamma1 = torch.reshape(torch.index_select(self.gamma2, 0, y), [-1, 1, 1, self.input_shape])
            print("BN y.shape =", y.shape, ", type(y) =", type(y))

            embedding_weight = nn.Embedding.from_pretrained(self.beta2)
            embedding_gamma = nn.Embedding.from_pretrained(self.gamma2)

            self.beta2 = torch.reshape(embedding_weight(y), [-1, self.input_shape])
            self.gamma2 = torch.reshape(embedding_gamma(y), [-1, self.input_shape])

        running_mean, running_var = torch.mean(x, dim=[0, 2, 3]), torch.var(x, dim=[0, 2, 3])

        print("God save me")
        print("x shape:", x.size())
        print("mean shape:", running_mean.size())
        print("var shape", running_var.size())
        print("beta shape", self.beta1.size())
        print("gamma shape", self.gamma1.size())

        normalized = F.batch_norm(x, running_mean, running_var, self.beta1, self.gamma1, eps=1e-3,
                                  momentum=0.5)

        return normalized


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


# class Downsampling(pl.LightningModule):
#     def __init__(self, kernel_size, stride):
#         super().__init__()

#     def forward(self, x):
#         dim_in = x.shape[2]
#         pad = dim_in//2 if dim_in % 2 == 0 else (dim_in + 1) // 2
#         return F.avg_pool2d(x, 2, 2, pad)


class Dense(pl.LightningModule):
    def __init__(self, in_channels, out_channels, is_sn=False):
        super().__init__()

        self.fc = nn.Linear(in_channels, out_channels)

        if is_sn:
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, x):
        x = self.fc(x)

        return x


# class InnerProduct(pl.LightningModule):
#     def __init__(self, in_channels, n_classes):
#         super().__init__()
#
#         self.v = nn.init.xavier_uniform_(torch.empty(size=[n_classes, in_channels]))
#
#     def forward(self, x, y):
#         self.v = self.v.transpose(0, 1)
#         self.v = nn.utils.spectral_norm(self.v).transpose(0, 1)
#
#         temp = nn.Embedding.from_pretrained(embeddings=self.v)(y)
#
#         # temp = torch.index_select(self.v, dim=0, index=y)
#         temp = torch.sum(temp * x, 1, keepdim=True)
#
#         return temp

class InnerProduct(pl.LightningModule):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        # nn.utils.spectral_norm inputs a layer, I wrapped v into Linear
        # I can not assign 2 spectral norm twice (python throws an Error, that's why I took it to __init__
        # self.dense = nn.Linear(in_channels, n_classes, bias=False)
        # nn.init.xavier_uniform_(self.dense.weight)
        # self.dense = nn.utils.spectral_norm(self.dense)

        self.embedding = torch.nn.Embedding(n_classes, in_channels)
        self.embedding = torch.nn.utils.spectral_norm(self.embedding)

    def forward(self, x, y):
        # print("Input inner product x shape:", x.size())

        # Cast y to long(), index should be int.
        temp = self.embedding(y.long())
        # print("temp size from index_select:", temp.size())

        x = x.squeeze()  # Сжимаем [n, 1024, 1, 1] до [n, 1024] и потом element-wise multiply with x

        # print("x shape:", x.size())
        # print("temp shape:", temp.size())

        temp = temp + x
        # print(temp.size(), "should be [n, 1024]")
        # print("temp size after sum:", temp.size())

        return temp


class GlobalSumPooling(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.sum(x, [1, 2])

        return x


class GeneratorEncoderResblock(pl.LightningModule):
    def get_gpu_memory(self):
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values[0]

    def __init__(self, in_channels, out_channels, is_sn=False):
        super().__init__()

        self.identity = nn.Identity()
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU()
        self.downsampling = Downsampling()
        self.conv1 = SpectralConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    is_sn=is_sn)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = SpectralConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    is_sn=is_sn)
        self.conv_identity = SpectralConv2d(in_channels, out_channels, kernel_size=1, stride=1, is_sn=is_sn)

    def forward(self, x):
        # print('\n\t\tGeneratorEncoderResblock')
        temp = self.identity(x)

        # print('\t\tFree GPU memory after  temp = self.identity(x): {} MB'.format(self.get_gpu_memory()))
        x = self.bn1(x)
        # print('\t\tFree GPU memory after  x = self.bn1(x): {} MB'.format(self.get_gpu_memory()))
        x = self.relu(x)
        # print('\t\tFree GPU memory after  x = self.relu(x): {} MB'.format(self.get_gpu_memory()))
        x = self.downsampling(x)
        # print('\t\tFree GPU memory after  x = self.downsampling(x): {} MB'.format(self.get_gpu_memory()))
        x = self.conv1(x)
        # print('\t\tFree GPU memory after  x = self.conv1(x): {} MB'.format(self.get_gpu_memory()))
        x = self.bn2(x)
        # print('\t\tFree GPU memory after  x = self.bn2(x): {} MB'.format(self.get_gpu_memory()))
        x = self.relu(x)
        # print('\t\tFree GPU memory after  x = self.relu(x): {} MB'.format(self.get_gpu_memory()))
        x = self.conv2(x)
        # print('\t\tFree GPU memory after  x = self.conv2(x): {} MB'.format(self.get_gpu_memory()))

        temp = self.downsampling(temp)
        # print('\t\tFree GPU memory after  temp = self.downsampling(temp): {} MB'.format(self.get_gpu_memory()))
        temp = self.conv_identity(temp)
        # print('\t\tFree GPU memory after  temp = self.conv_identity(temp): {} MB'.format(self.get_gpu_memory()))

        x += temp
        # print('\t\tFree GPU memory after  x += temp: {} MB'.format(self.get_gpu_memory()))

        return x


class GeneratorResblock(pl.LightningModule):
    def get_gpu_memory(self):
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values[0]

    def __init__(self, in_channels, out_channels, is_sn=False):
        super().__init__()

        self.identity = nn.Identity()
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU()
        self.upsampling = nn.Upsample(scale_factor=2)
        self.conv1 = SpectralConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    is_sn=is_sn)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = SpectralConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    is_sn=is_sn)
        self.conv_identity = SpectralConv2d(in_channels, out_channels, kernel_size=1, stride=1, is_sn=is_sn)

    def forward(self, x):
        # print('\n\t\tGeneratorResblock')

        temp = self.identity(x)
        # print('\t\tFree GPU memory after  temp = self.identity(x): {} MB'.format(self.get_gpu_memory()))

        x = self.bn1(x)
        # print('\t\tFree GPU memory after  x = self.bn1(x): {} MB'.format(self.get_gpu_memory()))
        x = self.relu(x)
        # print('\t\tFree GPU memory after  x = self.relu(x): {} MB'.format(self.get_gpu_memory()))
        x = self.upsampling(x)
        # print('\t\tFree GPU memory after  x = self.upsampling(x): {} MB'.format(self.get_gpu_memory()))
        x = self.conv1(x)
        # print('\t\tFree GPU memory after  x = self.conv1(x): {} MB'.format(self.get_gpu_memory()))
        x = self.bn2(x)
        # print('\t\tFree GPU memory after  x = self.bn2(x): {} MB'.format(self.get_gpu_memory()))
        x = self.relu(x)
        # print('\t\tFree GPU memory after  x = self.relu(x): {} MB'.format(self.get_gpu_memory()))
        x = self.conv2(x)
        # print('\t\tFree GPU memory after  x = self.conv2(x): {} MB'.format(self.get_gpu_memory()))

        temp = self.upsampling(temp)
        # print('\t\tFree GPU memory after  temp = self.upsampling(temp): {} MB'.format(self.get_gpu_memory()))
        temp = self.conv_identity(temp)
        # print('\t\tFree GPU memory after  temp = self.conv_identity(temp): {} MB'.format(self.get_gpu_memory()))

        x += temp
        # print('\t\tFree GPU memory after  x += temp: {} MB'.format(self.get_gpu_memory()))

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
