from modules.DenseNet import DenseNet121
from modules.DataModule import DataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/celebA_YSBBB_Classifier.yaml')
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join('checkpoints/classifier', config['name'], 'model.ckpt'),
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    logger = TensorBoardLogger(config['log_dir'], name=config['name'])

    data_module = DataModule(config)
    val_loader = data_module.val_dataloader()
    train_loader = data_module.train_dataloader()

    model = DenseNet121(config)

    trainer = pl.Trainer(gpus=1, max_epochs=model.epochs, logger=logger, checkpoint_callback=checkpoint_callback)

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
