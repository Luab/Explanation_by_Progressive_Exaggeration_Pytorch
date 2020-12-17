from collections import OrderedDict

import pytorch_lightning as pl
from explainer.Discriminator import Discriminator
from explainer.GeneratorEncoderDecoder import GeneratorEncoderDecoder
from classifier.DenseNet import DenseNet121
import torch
import torch.nn.functional as F
import numpy as np
import os
import subprocess as sp


#   References:
#   https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/models/gans/basic/basic_gan_module.py
class Explainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.train_step = 0
        self.val_step = 0
        
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.channels = config['num_channel']
        self.input_size = config['input_size']
        self.n_classes = config['num_class']
        self.n_bins = config['num_bins']
        self.target_class = config['target_class'] - 1
        self.lambda_gan = config['lambda_GAN']
        self.lambda_cyc = config['lambda_cyc']
        self.lambda_cls = config['lambda_cls']
        self.save_summary = int(config['save_summary'])
        self.ckpt_dir_continue = config['ckpt_dir_continue']

        self.G = GeneratorEncoderDecoder(self.n_bins)
        self.D = Discriminator(self.n_bins)

        self.model = DenseNet121(config, pretrained=False)
        cls_ckpt_path = os.path.join(config['cls_experiment'], 'last.ckpt')
        cls_ckpt = torch.load(cls_ckpt_path)
        self.model.load_state_dict(cls_ckpt['state_dict'])

        # Free memory that is reserved for gradients
        for p in self.model.parameters():
            p.requires_grad_(False)

    def get_gpu_memory(self):
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values[0]

    def forward(self, x, y):
        return self.G(x, y)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(params=self.G.parameters(), lr=0.0002, betas=(0., 0.9))
        d_opt = torch.optim.Adam(params=self.D.parameters(), lr=0.0002, betas=(0., 0.9))

        return [g_opt, d_opt], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_source, y_s = batch
        y_s = y_s.view(-1)
        y_s = self.convert_ordinal_to_binary(y_s, self.n_bins)

        y_t = torch.randint(low=0, high=self.n_bins, size=[self.batch_size])
        y_t = self.convert_ordinal_to_binary(y_t, self.n_bins)

        if batch_idx % 5 == 0 and optimizer_idx == 0:
            g_loss = self.generator_step(x_source, y_t, y_s)
            self.logger.experiment.add_scalar('g_loss_train', g_loss, self.train_step)
            return g_loss

        if optimizer_idx == 1:
            d_loss = self.discriminator_step(x_source, y_t, y_s)
            self.logger.experiment.add_scalar('d_loss_train', d_loss, self.train_step)
            return d_loss
        
        self.train_step += 1
        
        # Attention, we skip batch
        return None
    
    def validation_step(self, batch, batch_idx):
        x_source, y_s = batch
        y_s = y_s.view(-1)
        y_s = self.convert_ordinal_to_binary(y_s, self.n_bins)

        y_t = torch.randint(low=0, high=self.n_bins, size=[self.batch_size])
        y_t = self.convert_ordinal_to_binary(y_t, self.n_bins)

        g_loss = self.generator_step(x_source, y_t, y_s)
        self.logger.experiment.add_scalar('g_loss_val', g_loss, self.val_step)

        d_loss = self.discriminator_step(x_source, y_t, y_s)
        self.logger.experiment.add_scalar('d_loss_val', d_loss, self.val_step)
        
        self.val_step += 1
        
        return g_loss

    def generator_step(self, x_source, y_t, y_s):
        x_source, y_t, y_s = x_source.cuda(), y_t.cuda(), y_s.cuda()
        y_target = y_t[:, 0]
        y_source = y_s[:, 0]

        fake_target_img, fake_target_img_embedding = self.G(x_source, y_target)
        fake_source_recons_img, x_source_img_embedding = self.G(x_source, y_source)
        fake_source_img, fake_source_img_embedding = self.G(fake_target_img, y_source)

        g_loss_rec = F.mse_loss(x_source_img_embedding, fake_source_img_embedding)

        fake_target_logits = self.D(fake_target_img, y_t)
        g_loss_gan = self.generator_loss(fake_target_logits)
        g_loss_cyc = F.l1_loss(x_source, fake_source_img)

        fake_img_cls_logit_pretrained = self.model(fake_target_img)
        fake_evaluation = F.binary_cross_entropy_with_logits(
            y_target.clone().detach() * 0.1,
            fake_img_cls_logit_pretrained[:, self.target_class]
        )
        
        real_img_recons_cls_logit_pretrained = self.model(fake_source_img)
        real_img_cls_logits_pretrained = self.model(x_source)
        recons_evaluation = F.binary_cross_entropy_with_logits(
            real_img_recons_cls_logit_pretrained[:, self.target_class],
            real_img_cls_logits_pretrained[:, self.target_class],
        )

        g_loss = g_loss_gan * self.lambda_gan + \
                 (g_loss_rec + g_loss_cyc) * self.lambda_cyc + \
                 (fake_evaluation + recons_evaluation) * self.lambda_cls
        
        if self.val_step % 5 == 0:
            self.logger.experiment.add_image('True image', x_source[0], self.val_step)
            self.logger.experiment.add_image('Generated image', fake_target_img[0], self.val_step)

        return g_loss

    def discriminator_step(self, x_source, y_t, y_s):

        x_source, y_t, y_s = x_source.cuda(), y_t.cuda(), y_s.cuda()
        y_target = y_t[:, 0]

        real_source_logits = self.D(x_source, y_s)
        fake_target_img, fake_target_img_embedding = self(x_source, y_target)
        fake_target_logits = self.D(fake_target_img, y_t)

        d_loss_gan = self.discriminator_loss(real_source_logits, fake_target_logits)

        d_loss = d_loss_gan * self.lambda_gan

        return d_loss

    # There was a problem with device of tensor - now it is on cuda only
    def discriminator_loss(self, real, fake, loss_func=F.multilabel_margin_loss):

        if loss_func != F.multilabel_margin_loss:
            raise NotImplemented
            
        zero = torch.zeros(1, device='cuda')

        real_loss = - torch.min(-1.0 + real, zero).mean()
        fake_loss = - torch.min(-1.0 - fake, zero).mean()

        loss = real_loss + fake_loss

        return loss

    # There was a problem with device of tensor - now it is on cuda only
    def generator_loss(self, fake, loss_func=F.multilabel_margin_loss):

        if loss_func != F.multilabel_margin_loss:
            raise NotImplemented

        return - torch.mean(fake)

    # Instead of this we could use torch.nn.functional.one_hot, numpy is super slow
    @staticmethod
    def convert_ordinal_to_binary(y, n):
        y = y.int()
        new_y = torch.zeros([y.shape[-1], n])
        new_y[:, 0] = y
        for i in range(0, y.shape[0]):
            for j in range(1, y[i] + 1):
                new_y[i, j] = 1
        return new_y.float()
