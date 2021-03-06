import pytorch_lightning as pl
import torchvision
import torch
import torch.nn.functional as F
import os
from explainer.Discriminator import Discriminator
from explainer.GeneratorEncoderDecoder import GeneratorEncoderDecoder
from classifier.DenseNet import DenseNet121


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
        self.target_class = config['target_class']
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
        self.model.eval()

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
            g_loss = self.generator_step(x_source, y_t, y_s, 'train')
            self.log('train_g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return g_loss

        if optimizer_idx == 1:
            d_loss = self.discriminator_step(x_source, y_t, y_s, self.train_step, 'train')
            self.log('train_d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return d_loss

        self.train_step += 1
        
        return None

    def validation_step(self, batch, batch_idx):
        x_source, y_s = batch
        y_s = y_s.view(-1)
        y_s = self.convert_ordinal_to_binary(y_s, self.n_bins)

        y_t = torch.randint(low=0, high=self.n_bins, size=[self.batch_size])
        y_t = self.convert_ordinal_to_binary(y_t, self.n_bins)

        g_loss = self.generator_step(x_source, y_t, y_s, 'val')
        self.log('val_g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        d_loss = self.discriminator_step(x_source, y_t, y_s, self.val_step, 'val')
        self.log('val_d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.val_step += 1

        return g_loss

    def generator_step(self, x_source, y_t, y_s, stage):
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
        fake_img_cls_prediction = torch.sigmoid(fake_img_cls_logit_pretrained)

        real_p = y_target.clone().detach() * 0.1
        fake_q = fake_img_cls_prediction[:, 0]
        fake_evaluation = F.binary_cross_entropy(fake_q, real_p)
        fake_evaluation = -torch.mean(fake_evaluation)

        real_img_recons_cls_logit_pretrained = self.model(fake_source_img)
        real_img_recons_cls_prediction = torch.sigmoid(real_img_recons_cls_logit_pretrained)

        real_img_cls_logit_pretrained = self.model(x_source)
        real_img_cls_prediction = torch.sigmoid(real_img_cls_logit_pretrained)

        recons_evaluation = F.binary_cross_entropy(real_img_recons_cls_prediction[:, 0], real_img_cls_prediction[:, 0])
        recons_evaluation = -torch.mean(recons_evaluation)

        g_loss = g_loss_gan * self.lambda_gan + \
                 (g_loss_rec + g_loss_cyc) * self.lambda_cyc + \
                 (fake_evaluation + recons_evaluation) * self.lambda_cls
        
        if stage == 'train':
            if self.train_step % 500 == 0:
                grid_x_source = torchvision.utils.make_grid(x_source * 0.5 + 0.5, nrow=8)
                grid_fake_target_img = torchvision.utils.make_grid(fake_target_img * 0.5 + 0.5, nrow=8)
                grid_fake_source_img = torchvision.utils.make_grid(fake_source_img * 0.5 + 0.5, nrow=8)
                grid_fake_source_recons_img = torchvision.utils.make_grid(fake_source_recons_img * 0.5 + 0.5, nrow=8)
                self.logger.experiment.add_image('real_img : Train stage', grid_x_source, self.train_step)
                self.logger.experiment.add_image('fake_target_img : Train stage', grid_fake_target_img, self.train_step)
                self.logger.experiment.add_image('fake_source_img : Train stage', grid_fake_source_img, self.train_step)
                self.logger.experiment.add_image('fake_source_recons_img : Train stage', grid_fake_source_recons_img,
                                             self.train_step)
            
            self.log('train_g_loss_gan', g_loss_gan, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('train_g_loss_cyc', g_loss_cyc, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('train_g_loss_rec', g_loss_rec, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('train_fake_evaluation', fake_evaluation, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('train_recons_evaluation', recons_evaluation, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        if stage == 'val':
            if self.val_step % 500 == 0:
                grid_x_source = torchvision.utils.make_grid(x_source * 0.5 + 0.5, nrow=8)
                grid_fake_target_img = torchvision.utils.make_grid(fake_target_img * 0.5 + 0.5, nrow=8)
                grid_fake_source_img = torchvision.utils.make_grid(fake_source_img * 0.5 + 0.5, nrow=8)
                grid_fake_source_recons_img = torchvision.utils.make_grid(fake_source_recons_img * 0.5 + 0.5, nrow=8)
                self.logger.experiment.add_image('real_img : Validation stage', grid_x_source, self.val_step)
                self.logger.experiment.add_image('fake_target_img : Validation stage', grid_fake_target_img, self.val_step)
                self.logger.experiment.add_image('fake_source_img : Validation stage', grid_fake_source_img, self.val_step)
                self.logger.experiment.add_image('fake_source_recons_img : Validation stage', grid_fake_source_recons_img, self.val_step)
            
            self.log('val_g_loss_gan', g_loss_gan, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('val_g_loss_cyc', g_loss_cyc, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('val_g_loss_rec', g_loss_rec, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('val_fake_evaluation', fake_evaluation, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('val_recons_evaluation', recons_evaluation, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        return g_loss

    def discriminator_step(self, x_source, y_t, y_s, step, stage):
        x_source, y_t, y_s = x_source.cuda(), y_t.cuda(), y_s.cuda()
        y_target = y_t[:, 0]

        real_source_logits = self.D(x_source, y_s)
        fake_target_img, fake_target_img_embedding = self(x_source, y_target)
        fake_target_logits = self.D(fake_target_img, y_t)

        d_loss_gan = self.discriminator_loss(real_source_logits, fake_target_logits)

        if stage == 'train':
            self.log('train_d_loss_gan', d_loss_gan, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        if stage == 'val':
            self.log('val_d_loss_gan', d_loss_gan, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        d_loss = d_loss_gan * self.lambda_gan

        return d_loss

    def discriminator_loss(self, real, fake, loss_func=F.multilabel_margin_loss):
        if loss_func != F.multilabel_margin_loss:
            raise NotImplemented

        zero = torch.zeros(1, device='cuda')

        real_loss = -torch.mean(torch.min(-1.0 + real, zero))
        fake_loss = -torch.mean(torch.min(-1.0 - fake, zero))

        loss = real_loss + fake_loss

        return loss

    def generator_loss(self, fake, loss_func=F.multilabel_margin_loss):
        if loss_func != F.multilabel_margin_loss:
            raise NotImplemented

        return - torch.mean(fake)

    @staticmethod
    def convert_ordinal_to_binary(y, n):
        y = y.int()
        new_y = torch.zeros([y.shape[-1], n])
        new_y[:, 0] = y
        for i in range(0, y.shape[0]):
            for j in range(1, y[i] + 1):
                new_y[i, j] = 1
        return new_y.float()
      