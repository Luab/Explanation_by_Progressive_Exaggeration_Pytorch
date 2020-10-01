import pytorch_lightning as pl
from explainer.Discriminator import Discriminator
from explainer.GeneratorEncoderDecoder import GeneratorEncoderDecoder
from classifier.DenseNet import DenseNet121
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
        self.lambda_gan = config['lambda_GAN']
        self.lambda_cyc = config['lambda_cyc']
        self.lambda_cls = config['lambda_cls']
        self.save_summary = int(config['save_summary'])
        self.ckpt_dir_continue = config['ckpt_dir_continue']

        self.G = GeneratorEncoderDecoder()
        self.D = Discriminator()

        #   TODO make the model return logits and prediction
        self.model = DenseNet121(config)

    #   TODO implement Conditional Batch Normalization
    def forward(self, x, y):
        return self.G(x, y)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(params=self.G.parameters(), lr=0.0002, betas=(0., 0.9))
        d_opt = torch.optim.Adam(params=self.D.parameters(), lr=0.0002, betas=(0., 0.9))

        return [g_opt, d_opt], []

    #   TODO find out how to put in batch three variables: x_source, y_t, y_s
    def training_step(self, batch, batch_idx, optimizer_idx):
        x_source, y_t, y_s = batch
        y_target = y_t[:, 0]
        y_source = y_s[:, 0]

        result = None

        if optimizer_idx == 0:
            result = self.generator_step(x_source, y_target, y_source)

        if optimizer_idx == 1:
            result = self.discriminator_step(x_source, y_t, y_s)

        return result

    #   I suppose that we do not need validation_step
    def generator_step(self, x_source, y_target, y_source):

        #   We have to implement conditional batch norm!!!
        fake_target_img, fake_target_img_embedding = self(x_source, y_target)
        fake_source_recons_img, x_source_img_embedding = self(x_source, y_source)
        fake_source_img, fake_source_img_embedding = self(fake_target_img, y_source)

        g_loss_rec = F.mse_loss(x_source_img_embedding, fake_source_img_embedding)

        fake_target_logits = self.D(fake_target_img, y_target, self.n_bins)

        g_loss_gan = self.generator_loss(fake_target_logits)

        g_loss_cyc = F.l1_loss(x_source, fake_source_img)

        fake_img_cls_logit_pretrained, fake_img_cls_prediction = self.model(fake_target_img)
        fake_q = fake_img_cls_prediction[:, self.target_class]
        real_p = torch.tensor(y_target, dtype=torch.float32) * 0.1
        fake_evaluation = real_p * torch.log(fake_q) + (1 - real_p) * torch.log(1 - fake_q)

        real_img_recons_cls_logit_pretrained, real_img_recons_cls_prediction = self.model(fake_source_img)
        real_img_cls_logits_pretrained, real_img_cls_prediction = self.model(x_source)
        recons_evaluation = real_img_cls_prediction[:, self.target_class] * torch.log(real_img_recons_cls_prediction[:, self.target_class]) + (1 - real_img_recons_cls_prediction[:, self.target_class]) * torch.log(1 - real_img_recons_cls_prediction[:, self.target_class])
        recons_evaluation -= torch.mean(recons_evaluation)

        g_loss = g_loss_gan * self.lambda_gan + g_loss_rec * self.lambda_cyc + g_loss_cyc * self.lambda_cyc + fake_evaluation * self.lambda_cls + recons_evaluation * self.lambda_cls

        result = pl.TrainResult(minimize=g_loss, checkpoint_on=g_loss)
        result.log('g_loss', on_step=True, on_epoch=True, prog_bar=True)

        return result

    def discriminator_step(self, x_source, y_t, y_s):
        y_target = y_t[:, 0]

        real_source_logits = self.D(x_source, y_s, self.n_bins)
        fake_target_img, fake_target_img_embedding = self(x_source, y_target)
        fake_target_logits = self.D(fake_target_img, y_t, self.n_bins)

        d_loss_gan = self.discriminator_loss(real_source_logits, fake_target_logits)

        d_loss = d_loss_gan * self.lambda_gan

        result = pl.TrainResult(minimize=d_loss)
        result.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True)

        return result

    #   TODO check validity of loss functions for discriminator and generator
    #   Code is borrowed from here:
    #   https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/models/gans/basic/basic_gan_module.py#L10-L169
    #   x_real, x_fake are truth image and generated image respectively
    @staticmethod
    def discriminator_loss(self, real, fake, loss_func=F.multilabel_margin_loss):
        b = real.size(0)
        y_real = torch.ones(b, 1)

        #   x_real - logits, it has already been passed through forward step of D
        real_loss = F.binary_cross_entropy(real, y_real)

        y_fake = torch.zeros(b, 1)

        #   x_fake - logits, it has already been passed through forward step of D
        fake_loss = loss_func(fake, y_fake)

        # gradient backpropagation & optimize ONLY D's parameters
        loss = real_loss + fake_loss

        return loss

    @staticmethod
    def generator_loss(self, fake, loss_func=F.multilabel_margin_loss):
        fake_loss = 0

        if loss_func == F.multilabel_margin_loss:
            fake_loss -= torch.mean(fake)

        loss = fake_loss

        return loss
