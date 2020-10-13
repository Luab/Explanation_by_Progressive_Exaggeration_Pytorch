import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch


class ConditionalBatchNorm2d(pl.LightningModule):
    """
    One by one implementation of ConditionalBatchNorm from tf repo
    """

    class EMA:
        """
        Exponential moving average: we use it in order not to have big loss jumps
        Reference: https://en.wikipedia.org/wiki/Moving_average
        """
        def __init__(self):
            self.last_value = None
            self.coef = 0.5

        def __call__(self, x):
            if self.last_value is None:
                # May be detach is not needed, if we clone tensor
                self.last_value = x.clone().detach()
                return x
            else:
                temp = self.coef * x + (1 - self.coef) * self.last_value
                # May be detach is not needed, if we clone tensor
                self.last_value = temp.clone().detach()
                return temp

    def __init__(self, input_shape: int, num_classes: int):
        """
        Creates 2 type of variables for y is None and y is not None
        :param input_shape: it is x.shape[-1], we need to know it in at compile-time, I think we could guess it in debugging
        :param num_classes: NUmber of outputting features
        """
        super().__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.ema_mean = ConditionalBatchNorm2d.EMA()
        self.ema_var = ConditionalBatchNorm2d.EMA()

        self.beta1 = torch.zeros(size=[input_shape], requires_grad=True)
        self.gamma1 = torch.ones(size=[input_shape], requires_grad=True)

        self.beta2 = torch.zeros(size=[num_classes, input_shape], requires_grad=True)
        self.gamma2 = torch.ones(size=[num_classes, input_shape], requires_grad=True)

    def forward(self, x, y=None):
        if y is not None:
            self.beta1 = torch.reshape(torch.index_select(self.beta2, 0, y), [-1, 1, 1, self.input_shape])
            self.gamma1 = torch.reshape(torch.index_select(self.gamma2, 0, y), [-1, 1, 1, self.input_shape])

        batch_mean, batch_var = torch.mean(x, dim=[0, 2, 3], keepdim=True), torch.var(x, dim=[0, 2, 3], keepdim=True)

        mean, var = self.ema_mean(batch_mean), self.ema_var(batch_var)

        normalized = F.batch_norm(x, mean, var, self.beta1, self.gamma1, eps=1e-3)

        return normalized


class Upsampling(pl.LightningModule):
    def __init__(self, scale_factor):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor)

    def forward(self, x):
        x = self.upsampling(x)

        return x


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
        #x = F.pad(x, [padding_h, padding_h, padding_w, padding_w], 'constant', 0)
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


class InnerProduct(pl.LightningModule):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.v = nn.init.xavier_uniform_(torch.empty(size=[n_classes, in_channels]))

    def forward(self, x, y):
        self.v = self.v.transpose(0, 1)
        self.v = nn.utils.spectral_norm(self.v).transpose(0, 1)

        temp = nn.Embedding.from_pretrained(embeddings=self.v)(y)

        # temp = torch.index_select(self.v, dim=0, index=y)
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
        self.downsampling = Downsampling()
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
        self.upsampling = Upsampling(scale_factor=2)
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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        if is_sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        x = self.conv(x)

        return x
