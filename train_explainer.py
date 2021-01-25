import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from classifier.DataModule import DataModule
from explainer.Explainer import Explainer
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import yaml
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='./configs/celebA_Young_Explainer.yaml')
    parser.add_argument('--resume_from_ckpt', help='resume from the latest checkpoint', action='store_true')
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join('checkpoints/explainer', config['name'], 'exp'),
        save_last=True,
        save_top_k=1,
        monitor='val_g_loss',
        verbose=True,
        mode='min'
    )
    
    logger = TensorBoardLogger(config['log_dir'], name=config['name'])
    data_module = DataModule(config, to_explainer=True)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    explainer = Explainer(config)
    
    if args.resume_from_ckpt:
        print('Resuming from the latest checkpoint...')
        trainer = pl.Trainer(gpus=1, max_epochs=explainer.epochs, logger=logger, callbacks=[checkpoint_callback], accumulate_grad_batches=4, resume_from_checkpoint=os.path.join('checkpoints/explainer', config['name'], 'last.ckpt'))
    else:
        trainer = pl.Trainer(gpus=1, max_epochs=explainer.epochs, logger=logger, callbacks=[checkpoint_callback], accumulate_grad_batches=4)

    trainer.fit(explainer, train_loader, val_loader)


if __name__ == '__main__':
    main()
    