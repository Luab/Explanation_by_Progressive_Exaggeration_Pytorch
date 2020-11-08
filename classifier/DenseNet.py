import pytorch_lightning as pl
import torchvision.models as models
import torch.nn as nn
import torch


class DenseNet121(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.channels = config['num_channel']
        self.input_size = config['input_size']
        self.n_classes = config['num_class']

        self.model = models.densenet121(pretrained=True)
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
        result = pl.TrainResult(minimize=train_loss)
        print("Output size of DenseNet", outputs.size())
        result.log('train_loss', train_loss, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        validation_loss = self.loss(outputs, targets)
        result = pl.EvalResult(checkpoint_on=validation_loss)
        result.log('val_loss', validation_loss, on_step=True, on_epoch=True)
        return result
