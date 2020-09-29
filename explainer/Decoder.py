from explainer.utils import *
class Decoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.g_res_block1 = GeneratorResblock(in_channels=1024, out_channels=1024)
        self.g_res_block2 = GeneratorResblock(in_channels=1024, out_channels=512)
        self.g_res_block3 = GeneratorResblock(in_channels=512, out_channels=256)
        self.g_res_block4 = GeneratorResblock(in_channels=256, out_channels=128)
        self.g_res_block5 = GeneratorResblock(in_channels=128, out_channels=64)

        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv = SpectralConv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1)

        self.tanh = nn.Tanh()

    def forward(self, x):

        x = self.g_res_block1(x)
        x = self.g_res_block2(x)
        x = self.g_res_block3(x)
        x = self.g_res_block4(x)
        x = self.g_res_block5(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv(x)

        x = self.tanh(x)

        return x
