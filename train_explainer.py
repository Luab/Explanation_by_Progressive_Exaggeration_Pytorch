import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from classifier.DataModule import DataModule
from explainer.Explainer import Explainer
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import yaml
import os
import subprocess as sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/celebA_Young_Explainer.yaml')
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join('checkpoints/explainer', config['name'], 'explainer'),
        save_last=True,
        save_top_k=1,
        monitor='val_loss_abs',
        verbose=True,
        mode='min'
    )
    
    logger = TensorBoardLogger(config['log_dir'], name=config['name'])
    data_module = DataModule(config, from_explainer=True)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    explainer = Explainer(config)

    trainer = pl.Trainer(gpus=1, max_epochs=explainer.epochs, logger=logger, callbacks=[checkpoint_callback])

    trainer.fit(explainer, train_loader, val_loader)


if __name__ == '__main__':
    main()
