import pytorch_lightning as pl
import torch.nn as nn

from explainer.utils import *


class Discriminator(pl.LightningModule):
    def __init__(self, n_bins):
        super().__init__()

        self.n_bins = n_bins

        self.d_res_block1 = DiscriminatorResBlock(in_channels=3, out_channels=64, is_first=True)
        self.d_res_block2 = DiscriminatorResBlock(in_channels=64, out_channels=128)
        self.d_res_block3 = DiscriminatorResBlock(in_channels=128, out_channels=256)
        self.d_res_block4 = DiscriminatorResBlock(in_channels=256, out_channels=512)
        self.d_res_block5 = DiscriminatorResBlock(in_channels=512, out_channels=1024)
        self.d_res_block6 = DiscriminatorResBlock(in_channels=1024, out_channels=1024, is_down=False)

        self.relu = nn.ReLU()
        self.global_sum_pooling = nn.AvgPool2d(kernel_size=4)

        self.inner_product = nn.utils.spectral_norm(
            nn.Embedding(n_bins, 1024)
        )
        self.fc = nn.utils.spectral_norm(nn.Linear(1024, 1))

    def initialise_(self):
        """xavier initialization for self.inner_product"""
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y):
        y = y[:, 0].squeeze().long()  # TODO Delete it after torchsummary
        # print('shape of y:', y.size())
        x = self.d_res_block1(x)
        x = self.d_res_block2(x)
        x = self.d_res_block3(x)
        x = self.d_res_block4(x)
        x = self.d_res_block5(x)
        x = self.d_res_block6(x)
        x = self.relu(x)
        # x = self.global_sum_pooling(x)
        x = torch.sum(x, dim=(2, 3))
        # print('shape of x:', x.size())

        outputs = self.fc(x)

        # Code was taken from
        # https://github.com/crcrpar/pytorch.sngan_projection/blob/master/models/discriminators/snresnet64.py
        if y is not None:
            outputs += torch.sum(self.inner_product(y) * x, dim=1, keepdim=True)

        return outputs
