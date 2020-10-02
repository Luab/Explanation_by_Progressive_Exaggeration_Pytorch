import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from classifier.DataModule import DataModule
from explainer.Explainer import Explainer
import argparse
import yaml


#   TODO synchronize path with saved models with path for saving in config file
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/celebA_Young_Explainer.yaml')
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger = TensorBoardLogger(config['log_dir'], name=config['name'])
    data_module = DataModule(config)
    val_loader = data_module.val_dataloader()
    train_loader = data_module.train_dataloader()

    explainer = Explainer(config, logger)

    trainer = pl.Trainer(gpus=1, max_epochs=explainer.epochs)

    trainer.fit(explainer, train_loader)


if __name__ == '__main__':
    main()
