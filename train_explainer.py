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
    parser.add_argument('--config', '-c', default='/home/intern/BS19_implementation/configs/celebA_Young_Explainer.yaml')
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
    # Let's for now skip logger it until training will start
    # logger = TensorBoardLogger(config['log_dir'], name=config['name'])
    data_module = DataModule(config, from_explainer=True)
    train_loader = data_module.train_dataloader()

    # In Explainer we shouldn't pass logger, only into Trainer
    explainer = Explainer(config)

    # Let's for now skip logger it until training will start
    # By setting val_percent_check=0.0 validation would not execute
    trainer = pl.Trainer(gpus=1, max_epochs=explainer.epochs, val_check_interval=0.0)

    trainer.fit(explainer, train_loader)


if __name__ == '__main__':
    main()
