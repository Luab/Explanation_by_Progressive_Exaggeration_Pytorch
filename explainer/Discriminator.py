import pytorch_lightning as pl
from explainer.utils import *
import torch.nn as nn


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
        self.inner_product = InnerProduct(in_channels=1024, n_classes=2)
        self.fc = nn.utils.spectral_norm(nn.Linear(1024, 1))
    
     #  Here I deleted y argument in def forward
     #  TODO need to repair! add y argument (and fix inner-product function) and evaluate the total size
    def forward(self, x):
        x = self.d_res_block1(x)
        x = self.d_res_block2(x)
        x = self.d_res_block3(x)
        x = self.d_res_block4(x)
        x = self.d_res_block5(x)
        x = self.d_res_block6(x)

        x = self.relu(x)

        x = self.global_sum_pooling(x)

        # for i in range(0, self.n_bins - 1):
            # if i == 0:
                # temp = self.inner_product(x, y[:, i + 1])
            # else:
                # temp += self.inner_product(x, y[:, i + 1])

        x = x.view(-1, 1024)
        x = self.fc(x)

        # x += temp

        return x
