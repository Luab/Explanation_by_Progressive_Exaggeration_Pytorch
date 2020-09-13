import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.models as models
import torch.nn as nn
import torch
import yaml
import argparse
from utils import read_data_file, load_images_and_labels
import pandas as pd
import sys
import os
import pdb


%load_ext tensorboard
%tensorboard --logdir logs/

logger = TensorBoardLogger('logs', name='densenet121')
trainer = pl.Trainer(gpus=1, max_nb_epochs=model.epochs, logger=logger)
model = DenseNet121()
trainer.fit(model)


class DenseNet121(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.parse_arguments()
        self.load_config()
        self.set_experiment_folder()
        self.set_parameters()
        self.load_data()
        self.is_train = True
        self.create_model()
        self.loss = nn.BCEWithLogitsLoss()
        self.current_epoch = 0


    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', '-c', default='configs/celebA_YSBBB_Classifier.yaml')
        self.args = parser.parse_args()


    def load_config(self):
        self.config_path = self.args.config
        self.config = yaml.load(open(self.config_path))
        print(self.config)


    def set_experiment_folder(self):
        self.output_dir = os.path.join(self.config['log_dir'], self.config['name'])

        try:
            os.makedirs(self.output_dir)
        except:
            pass
        try:
            os.makedirs(os.path.join(self.output_dir, 'logs'))
        except:
            pass


    def set_parameters(self):
        self.batch_size = self.config['batch_size']
        self.epochs = self.config['epochs']
        self.channels = self.config['num_channel']
        self.input_size = self.config['input_size']
        self.n_classes = self.config['num_class']
        self.ckpt_dir_continue = self.config['ckpt_dir_continue']

        if self.ckpt_dir_continue == '':
            self.continue_train = False
        else:
            self.continue_train = True


    def load_data(self):
        try:
            categories, self.file_names_dict = read_data_file(config['image_label_dict'])
        except:
            print("Problem in reading input data file : ", self.config['image_label_dict'])
            sys.exit()

        self.data_train = np.load(self.config['train'])
        self.data_test = np.load(self.config['test'])

        print("The classification categories are: ")
        print(categories)

        print('The size of the training set: ', self.data_train.shape[0])
        print('The size of the testing set: ', self.data_test.shape[0])

        fp = open(os.path.join(self.output_dir, 'setting.txt'), 'w')
        fp.write('config_file:' + str(self.config_path) + '\n')
        fp.close()


    def create_model(self):
        self.model = models.densenet121(pretrained=True)
        num_ftrs = self.mdoel.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.n_classes)


    def forward(self, x):
        return self.model(x)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        return optimizer


    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        train_loss = self.loss(outputs, targets)
        return {'train_loss': train_loss}


    def training_epoch_end(self, training_step_outputs):
        train_loss_avg = torch.stack([x['train_loss'] for x in training_step_outputs]).mean()

        self.current_epoch += 1

        self.logger.experiment.add_scalar('Loss : training stage', train_loss_avg, self.current_epoch)

        return {'train_loss_avg': train_loss_avg}


    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        validation_loss = self.loss(outputs, targets)
        return {'validation_loss': validation_loss}


    def validation_epoch_end(self, validation_step_outputs):
        validation_loss_avg = torch.stack([x['validation_loss'] for x in validation_step_outputs]).mean()

        self.logger.experiment.add_scalar('Loss : training stage', validation_loss_avg, self.current_epoch)

        return {'validation_loss_avg': validation_loss_avg}
