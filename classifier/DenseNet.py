import pytorch_lightning as pl
import torchvision.models as models
import torch.nn as nn
import torch


class DenseNet121(pl.LightningModule):
    def __init__(self, config, pretrained):
        super().__init__()

        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.channels = config['num_channel']
        self.input_size = config['input_size']
        self.n_classes = config['num_class']
        self.model = models.densenet121(pretrained=pretrained)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, self.n_classes)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        train_loss = self.loss(outputs, targets)

        self.log('train_loss', train_loss, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        val_loss = self.loss(outputs, targets)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True)
        return val_loss
