import pytorch_lightning as pl
from explainer.Discriminator import Discriminator
from explainer.GeneratorEncoderDecoder import GeneratorEncoderDecoder
from explainer.utils import DiscriminatorLoss, GeneratorLoss


class Explainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.ckpt_dir_cls = config['cls_experiment']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.channels = config['num_channel']
        self.input_size = config['input_size']
        self.n_classes = config['num_class']
        self.n_bins = config['num_bins']
        self.target_class = config['target_class']
        self.lambda_GAN = config['lambda_GAN']
        self.lambda_cyc = config['lambda_cyc']
        self.lambda_cls = config['lambda_cls']
        self.save_summary = int(config['save_summary'])
        self.ckpt_dir_continue = config['ckpt_dir_continue']

        self.G = GeneratorEncoderDecoder()
        self.D = Discriminator()

        g_loss_gan = GeneratorLoss()
        d_loss_gan = DiscriminatorLoss()

    def forward(self):
        pass





