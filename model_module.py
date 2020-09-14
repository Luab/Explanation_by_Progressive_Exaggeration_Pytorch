import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.models as models
import torch.nn as nn
import torch
import yaml
import argparse
from data_module import DataModule


class DenseNet121(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #self.parse_arguments()
        self.load_config()
        self.set_parameters()
        self.create_model()
        self.loss = nn.BCEWithLogitsLoss()
        self.current_epoch = 0


    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', '-c', default='configs/celebA_YSBBB_Classifier.yaml')
        self.args = parser.parse_args()


    def load_config(self):
        #self.config_path = self.args.config
        self.config_path = 'configs/celebA_YSBBB_Classifier.yaml'
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        print(self.config)


    def set_parameters(self):
        self.batch_size = self.config['batch_size']
        self.epochs = self.config['epochs']
        self.channels = self.config['num_channel']
        self.input_size = self.config['input_size']
        self.n_classes = self.config['num_class']


    def create_model(self):
        self.model = models.densenet121(pretrained=True)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, self.n_classes)


    def forward(self, x):
        return self.model(x)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        return optimizer


    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        train_loss = self.loss(outputs, targets)
        return {'loss': train_loss}


    def training_epoch_end(self, training_step_outputs):
        train_loss_avg = torch.stack([x['loss'] for x in training_step_outputs]).mean()

        self.current_epoch += 1

        self.logger.experiment.add_scalar('Loss : training stage', train_loss_avg, self.current_epoch)

        return {'train_loss_avg': train_loss_avg}


    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        validation_loss = self.loss(outputs, targets)
        return {'loss': validation_loss}


    def validation_epoch_end(self, validation_step_outputs):
        validation_loss_avg = torch.stack([x['loss'] for x in validation_step_outputs]).mean()

        self.logger.experiment.add_scalar('Loss : training stage', validation_loss_avg, self.current_epoch)

        return {'validation_loss_avg': validation_loss_avg}

