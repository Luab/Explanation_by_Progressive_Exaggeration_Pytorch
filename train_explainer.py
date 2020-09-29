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
        self.channels = config['channels']
        self.lambda_GAN = config['lambda_GAN']
        self.num_class = config['classes']
        self.num_binary_outputs = config['num_bins']
        self.target_class = config['target_class']


        # Turn off gradients
        self.pretrained_classifier = pretrained_classifier
        self.pretrained_classifier.freeze()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()



    def configure_optimizers(self):
        # We have separated classes into 2 separate files,
        # need to think how we will calculate loss for them
        discriminator_optim = torch.optim.Adam(self.discriminator.parameters(), 2e-4, betas=(0.0, 0.9))
        encoder_optim = torch.optim.Adam(self.encoder.parameters(), 2e-4, betas=(0.0, 0.9))
        decoder_optim = torch.optim.Adam(self.decoder.parameters(), 2e-4, betas=(0.0, 0.9))

        return [discriminator_optim, encoder_optim, decoder_optim]

    def training_step(self, batch, batch_idx):
        # Need to think, how many outputs classes we have in pretrained_classifier
        #
        x_source, y_target = batch
        fake_target_img_embedding = self.encoder(x_source)
        fake_target_img = self.decoder(fake_target_img_embedding)
        fake_source_img_embedding = self.encoder(fake_target_img)
        fake_source_img = self.decoder(fake_source_img_embedding)

        fake_target_logits = self.discriminator(fake_target_img)

        real_img_cls_logit_pretrained, real_ing_cls_prediction = self.pretrained_classifier(x_source)
        fake_img_cls_logit_pretrained, fake_img_cls_prediction = self.pretrained_classifier(fake_target_img)
        real_img_recons_cls_logit_pretrained, real_img_recons_cls_prediction = self.pretrained_classifier(fake_source_img)


        real_p = y_target * 0.1
        fake_q = fake_img_cls_prediction[:, self.target_class]
        fake_evaluation = (real_p * torch.tan(fake_q)) + ( (1 - real_p) * torch.log(1 - fake_q))
        fake_evaluation = -torch.sum(fake_evaluation)











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