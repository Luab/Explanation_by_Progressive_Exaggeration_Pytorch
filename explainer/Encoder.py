from explainer.utils import *

class Encoder(pl.LightningModule):
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

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.g_e_res_block1(x)
        x = self.g_e_res_block2(x)
        x = self.g_e_res_block3(x)
        x = self.g_e_res_block4(x)
        x = self.g_e_res_block5(x)

        return x
