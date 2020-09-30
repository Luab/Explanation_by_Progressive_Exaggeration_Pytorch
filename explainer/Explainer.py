import pytorch_lightning as pl
from explainer.Discriminator import Discriminator
from explainer.GeneratorEncoderDecoder import GeneratorEncoderDecoder
import torch
import torch.nn.functional as F


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

    def forward(self):
        pass

    #   Code is borrowed from here:
    #   https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/models/gans/basic/basic_gan_module.py#L10-L169
    #   x_real, x_fake are truth image and generated image respectively
    def discriminator_loss(self, x_real, x_fake, loss_func=F.multilabel_margin_loss):
        b = x_real.size(0)
        x_real = x_real.view(b, -1)
        y_real = torch.ones(b, 1)

        #   calculate real score
        output = self.D(x_real)
        real_loss = F.binary_cross_entropy(output, y_real)

        y_fake = torch.zeros(b, 1)

        # calculate fake score
        output = self.D(x_fake)
        fake_loss = loss_func(output, y_fake)

        # gradient backpropagation & optimize ONLY D's parameters
        loss = real_loss + fake_loss

        return loss

    def generator_loss(self, x_fake, loss_func=F.multilabel_margin_loss):
        y_fake = torch.ones(x_fake.size(0), 1)

        # calculate fake score
        output = self.D(x_fake)
        loss = loss_func(output, y_fake)

        return loss


