import pytorch_lightning as pl
from explainer.utils import *
import subprocess as sp
import torch.nn as nn


class GeneratorEncoderDecoder(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.relu = nn.ReLU()
        self.bn1 = ConditionalBatchNorm2d(input_shape=3, num_classes=num_classes)
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

        self.bn2 = ConditionalBatchNorm2d(input_shape=64, num_classes=num_classes)
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
        print("\n\tGeneratorEncoderDecoder")
        print('\tFree GPU memory before forward propagation: {} MB'.format(self.get_gpu_memory()))
        x = self.bn1(x, y)
        print('\tFree GPU memory after  x = self.bn1(x): {} MB'.format(self.get_gpu_memory()))
        x = self.relu(x)
        print('\tFree GPU memory after  x = self.relu(x): {} MB'.format(self.get_gpu_memory()))
        x = self.conv1(x)
        print('\tFree GPU memory after  x = self.conv1(x): {} MB'.format(self.get_gpu_memory()))
        x = self.g_e_res_block1(x)
        print('\tFree GPU memory after  x = self.g_e_res_block1(x): {} MB'.format(self.get_gpu_memory()))
        x = self.g_e_res_block2(x)
        print('\tFree GPU memory after  x = self.g_e_res_block2(x): {} MB'.format(self.get_gpu_memory()))
        x = self.g_e_res_block3(x)
        print('\tFree GPU memory after  x = self.g_e_res_block3(x): {} MB'.format(self.get_gpu_memory()))
        x = self.g_e_res_block4(x)
        print('\tFree GPU memory after  x = self.g_e_res_block4(x): {} MB'.format(self.get_gpu_memory()))
        embedding = self.g_e_res_block5(x)
        print('\tFree GPU memory after embedding = self.g_e_res_block5(x): {} MB'.format(self.get_gpu_memory()))
        
        x = self.g_res_block1(embedding)
        print('\tFree GPU memory after  x = self.g_res_block1(embedding): {} MB'.format(self.get_gpu_memory()))
        x = self.g_res_block2(x)
        print('\tFree GPU memory after  x = self.g_res_block2(x): {} MB'.format(self.get_gpu_memory()))
        x = self.g_res_block3(x)
        print('\tFree GPU memory after  x = self.g_res_block3(x): {} MB'.format(self.get_gpu_memory()))
        x = self.g_res_block4(x)
        print('\tFree GPU memory after  x = self.g_res_block4(x): {} MB'.format(self.get_gpu_memory()))
        x = self.g_res_block5(x)
        print('\tFree GPU memory after  x = self.g_res_block5(x): {} MB'.format(self.get_gpu_memory()))

        x = self.bn2(x, y)
        print('\tFree GPU memory after  x = self.bn2(x): {} MB'.format(self.get_gpu_memory()))
        x = self.relu(x)
        print('\tFree GPU memory after  x = self.relu(x): {} MB'.format(self.get_gpu_memory()))
        x = self.conv(x)
        print('\tFree GPU memory after  x = self.conv(x): {} MB'.format(self.get_gpu_memory()))

        x = self.tanh(x)
        print('\tFree GPU memory after  x = self.tanh(x): {} MB'.format(self.get_gpu_memory()))
        
        return x, embedding
