import pytorch_lightning as pl
from explainer.utils import *
import torch.nn as nn


class GeneratorEncoderDecoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=3)
        self.conv1 = SpectralConv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1)

        self.g_e_res_block1 = GeneratorEncoderResblock(in_channels=64, out_channels=128)
        self.g_e_res_block2 = GeneratorEncoderResblock(in_channels=128, out_channels=256)
        self.g_e_res_block3 = GeneratorEncoderResblock(in_channels=256, out_channels=512)
        self.g_e_res_block4 = GeneratorEncoderResblock(in_channels=512, out_channels=1024)
        self.g_e_res_block5 = GeneratorEncoderResblock(in_channels=1024, out_channels=1024)

        self.g_res_block1 = GeneratorResblock(in_channels=1024, out_channels=1024)
        self.g_res_block2 = GeneratorResblock(in_channels=1024, out_channels=512)
        self.g_res_block3 = GeneratorResblock(in_channels=512, out_channels=256)
        self.g_res_block4 = GeneratorResblock(in_channels=256, out_channels=128)
        self.g_res_block5 = GeneratorResblock(in_channels=128, out_channels=64)

        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv = SpectralConv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1)

        self.tanh = nn.Tanh()

    def forward(self, x, y):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.g_e_res_block1(x)
        x = self.g_e_res_block2(x)
        x = self.g_e_res_block3(x)
        x = self.g_e_res_block4(x)
        embedding = self.g_e_res_block5(x)
        
        x = self.g_res_block1(x)
        x = self.g_res_block2(x)
        x = self.g_res_block3(x)
        x = self.g_res_block4(x)
        x = self.g_res_block5(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv(x)

        x = self.tanh(x)

        return x, embedding
