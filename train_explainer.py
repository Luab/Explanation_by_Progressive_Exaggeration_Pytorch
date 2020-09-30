import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import argparse
import yaml
from explainer.Decoder import Decoder
from explainer.Encoder import Encoder
from explainer.Discriminator import Discriminator
import torch
import torch.nn as nn


class Explainer(pl.LightningModule):
    def __init__(self, config: dict, pretrained_classifier: pl.LightningModule):
        super().__init__()
        self.channels = config['channels']
        self.lambda_GAN = config['lambda_GAN']
        self.num_class = config['classes']
        self.num_binary_outputs = config['num_bins']
        self.target_class = config['target_class']

        # Lambdas for loss functions
        self.lambda_GAN = config['lambda_GAN']
        self.lambda_cyc = config['lambda_cyc']
        self.lambda_cls = config['lambda_cls']

        # Turn off gradients
        self.pretrained_classifier = pretrained_classifier
        self.pretrained_classifier.freeze()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()

    def configure_optimizers(self):
        discriminator_optim = torch.optim.Adam(self.discriminator.parameters(), 2e-4, betas=(0.0, 0.9))
        # We need to connect encoder AND decoder parameters to be updated by generator_optim
        generator_optim = torch.optim.Adam(self.encoder.parameters(), 2e-4, betas=(0.0, 0.9))

        return [discriminator_optim, generator_optim]

    def training_step(self, batch, batch_idx):
        # Need to think, how many outputs classes we have in pretrained_classifier
        # I don't understand where there here delta's
        # y_target in source code - random array with labels, y_source - actual labels, why we passes it in Generator - Idk
        x_source, y_target = batch

        # ============= G & D =============
        real_source_logits = self.discriminator(x_source)

        fake_target_img_embedding = self.encoder(x_source)
        fake_target_img = self.decoder(fake_target_img_embedding)
        fake_source_img_embedding = self.encoder(fake_target_img)
        fake_source_img = self.decoder(fake_source_img_embedding)
        x_source_img_embedding = self.encoder(x_source)
        fake_source_recons_img = self.decoder(x_source_img_embedding)

        fake_target_logits = self.discriminator(fake_target_img)

        real_img_cls_logit_pretrained, real_img_cls_prediction = self.pretrained_classifier(x_source)
        fake_img_cls_logit_pretrained, fake_img_cls_prediction = self.pretrained_classifier(fake_target_img)
        real_img_recons_cls_logit_pretrained, real_img_recons_cls_prediction = self.pretrained_classifier(
            fake_source_img)

        fake_evaluation_loss = nn.BCELoss()(y_target * 0.1, fake_img_cls_prediction[:, self.target_class])

        recons_evaluation_loss = nn.BCELoss()(real_img_cls_prediction[:, self.target_class],
                                              real_img_recons_cls_prediction[:, self.target_class])
        D_loss_GAN = \
            -torch.mean(torch.min(torch.tensor([0.0]), -1.0 + real_source_logits)) \
            - torch.mean(torch.min(torch.tensor([0.0]), -1.0 - fake_target_logits))

        G_loss_GAN = - torch.mean(fake_target_logits)
        G_loss_cyc = nn.L1loss()(x_source, fake_source_img)
        G_loss_rec = nn.MSELoss()(x_source_img_embedding, fake_source_img_embedding)

        # G_loss and D_loss we want to optimize Adam
        G_loss = G_loss_GAN * self.lambda_GAN + (G_loss_cyc + G_loss_rec) * self.lambda_cyc + \
                 (fake_evaluation_loss + recons_evaluation_loss) * self.lambda_cls

        D_loss = D_loss_GAN * self.lambda_GAN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', '-c', default='configs/celebA_Young_Explainer.yaml')
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)


if __name__ == '__main__':
    main()
