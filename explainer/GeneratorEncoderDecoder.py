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
        print("Encoder")
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        print(f"x shape: {x.size()}, and should be [n, 64, 128, 128]")
        x = self.g_e_res_block1(x)
        print(f"x shape: {x.size()}, and should be [n, 128, 64, 64]")
        x = self.g_e_res_block2(x)
        print(f"x shape: {x.size()}, and should be [n, 256, 32, 32]")
        x = self.g_e_res_block3(x)
        print(f"x shape: {x.size()}, and should be [n, 512, 16, 16]")
        x = self.g_e_res_block4(x)
        print(f"x shape: {x.size()}, and should be [n, 1024, 8, 8]")
        embedding = self.g_e_res_block5(x)
        print(f"embedding shape: {embedding.size()}, and should be [n, 1024, 4, 4]")
        
        print("Decoder")

        # In original article we use embedding
        x = self.g_res_block1(embedding)
        print(f"x shape: {x.size()}, but should be [n, 1024, 8, 8]")
        x = self.g_res_block2(x)
        print(f"x shape: {x.size()}, but should be [n, 512, 16, 16]")
        x = self.g_res_block3(x)
        print(f"x shape: {x.size()}, but should be [n, 256, 32, 32]")
        x = self.g_res_block4(x)
        print(f"x shape: {x.size()}, but should be [n, 128, 64, 64]")
        x = self.g_res_block5(x)
        print(f"x shape: {x.size()}, but should be [n, 64, 128, 128]")

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv(x)
        print(f"x shape: {x.size()}, but should be [n, 3, 128, 128]")

        x = self.tanh(x)

        return x, embedding
