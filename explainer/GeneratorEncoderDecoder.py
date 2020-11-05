import pytorch_lightning as pl
from explainer.utils import *
import subprocess as sp
import torch.nn as nn


class GeneratorEncoderDecoder(pl.LightningModule):
    def __init__(self, n_bins):
        super().__init__()

        self.relu = nn.ReLU()
        self.bn1 = ConditionalBatchNorm2d(num_features=3, num_classes=n_bins)
        self.conv1 = SpectralConv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1)

        self.g_e_res_block1 = GeneratorEncoderResblock(in_channels=64, out_channels=128, num_classes=n_bins)
        self.g_e_res_block2 = GeneratorEncoderResblock(in_channels=128, out_channels=256, num_classes=n_bins)
        self.g_e_res_block3 = GeneratorEncoderResblock(in_channels=256, out_channels=512, num_classes=n_bins)
        self.g_e_res_block4 = GeneratorEncoderResblock(in_channels=512, out_channels=1024, num_classes=n_bins)
        self.g_e_res_block5 = GeneratorEncoderResblock(in_channels=1024, out_channels=1024, num_classes=n_bins)

        self.g_res_block1 = GeneratorResblock(in_channels=1024, out_channels=1024, num_classes=n_bins)
        self.g_res_block2 = GeneratorResblock(in_channels=1024, out_channels=512, num_classes=n_bins)
        self.g_res_block3 = GeneratorResblock(in_channels=512, out_channels=256, num_classes=n_bins)
        self.g_res_block4 = GeneratorResblock(in_channels=256, out_channels=128, num_classes=n_bins)
        self.g_res_block5 = GeneratorResblock(in_channels=128, out_channels=64, num_classes=n_bins)

        self.bn2 = ConditionalBatchNorm2d(num_features=64, num_classes=n_bins)
        self.conv = SpectralConv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1)

        self.tanh = nn.Tanh()
    
    def get_gpu_memory(self):
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]

        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values[0]
    
    def forward(self, x, y):
        y = y.squeeze().long()  # TODO delete it after summary

        print('In Generator, x-shape: {}'.format(x.size()))
        print('In Generator, y-shape: {}'.format(y.size()))
        x = self.bn1(x, y)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.g_e_res_block1(x, y)
        x = self.g_e_res_block2(x, y)
        x = self.g_e_res_block3(x, y)
        x = self.g_e_res_block4(x, y)
        embedding = self.g_e_res_block5(x, y)

        x = self.g_res_block1(embedding, y)
        x = self.g_res_block2(x, y)
        x = self.g_res_block3(x, y)
        x = self.g_res_block4(x, y)
        x = self.g_res_block5(x, y)

        x = self.bn2(x, y)
        x = self.relu(x)
        x = self.conv(x)
        x = self.tanh(x)
        
        return x, embedding
