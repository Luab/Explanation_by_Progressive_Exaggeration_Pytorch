import pytorch_lightning as pl
from explainer.utils import *
import torch.nn as nn


class Discriminator(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.d_res_block1 = DiscriminatorResBlock(in_channels=3, out_channels=64)
        self.d_res_block2 = DiscriminatorResBlock(in_channels=64, out_channels=128)
        self.d_res_block3 = DiscriminatorResBlock(in_channels=128, out_channels=256)
        self.d_res_block4 = DiscriminatorResBlock(in_channels=256, out_channels=512)
        self.d_res_block5 = DiscriminatorResBlock(in_channels=512, out_channels=1024, is_down=False)
        self.relu = nn.ReLU()
        self.global_sum_pooling = GlobalSumPooling()
        self.inner_product = InnerProduct(in_channels=1024, n_classes=2)
        self.dense = Dense(in_channels=1024, out_channels=1, is_sn=True)

    def forward(self, x, y, n_classes):
        x = self.d_res_block1(x)
        x = self.d_res_block2(x)
        x = self.d_res_block3(x)
        x = self.d_res_block4(x)
        x = self.d_res_block5(x)

        x = self.relu(x)

        x = self.global_sum_pooling(x)

        for i in range(0, n_classes - 1):
            if i == 0:
                temp = self.inner_product(x, y[:, i + 1], 2)
            else:
                temp += self.inner_product(x, y[:, i + 1], 2)

        x = self.dense(x, 1, is_sn=True)
        x += temp

        return x
