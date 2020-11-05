import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from classifier.DataModule import DataModule
from explainer.Explainer import Explainer
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import yaml
import os
import subprocess as sp

#   TODO synchronize path with saved models with path for saving in config file
def main():
    # print('***************************')
    # print(get_gpu_memory())
    # print('***************************')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/celebA_Young_Explainer.yaml')
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # They simply save each 500 iterations, but I monitor only min g_loss
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join('checkpoints/explainer', config['name'], 'model.ckpt'),
        save_last=True,
        save_top_k=1,
        monitor='g_loss',
        verbose=True,
        mode='min',
    )
    logger = TensorBoardLogger(config['log_dir'], name=config['name'])
    data_module = DataModule(config, from_explainer=True)
    val_loader = data_module.val_dataloader()
    train_loader = data_module.train_dataloader()

    # In Explainer we shouldn't pass logger, only into Trainer
    explainer = Explainer(config)

    trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=explainer.epochs, checkpoint_callback=checkpoint_callback)

    trainer.fit(explainer, train_loader, val_loader)


if __name__ == '__main__':
    main()
