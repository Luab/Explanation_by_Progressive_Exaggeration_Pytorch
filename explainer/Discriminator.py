import pytorch_lightning as pl
from explainer.utils import *
import torch.nn as nn


class Discriminator(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, n_classes):
        x = DiscriminatorResBlock(in_channels=3, out_channels=64)(x)
        x = DiscriminatorResBlock(in_channels=64, out_channels=128)(x)
        x = DiscriminatorResBlock(in_channels=128, out_channels=256)(x)
        x = DiscriminatorResBlock(in_channels=256, out_channels=512)(x)
        x = DiscriminatorResBlock(in_channels=512, out_channels=1024, is_down=False)(x)

        x = nn.ReLU()(x)

        x = GlobalSumPooling()(x)

        for i in range(0, n_classes - 1):
            if i == 0:
                temp = InnerProduct(x, y[:, i + 1], 2)
            else:
                temp += InnerProduct(x, y[:, i + 1], 2)

        x = Dense()(x, 1, is_sn=True)

        x += temp

        return x
