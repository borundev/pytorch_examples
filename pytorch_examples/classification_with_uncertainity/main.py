import os
import sys

import names
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

from data.fudge_labels import modify_data_module
from data.kaggle import TBDataModule
from pytorch_examples.transfer_learning.model import CustomModel

CustomModel.make_epoch_end_funcs(['Normal', 'TB'])


if 'KAGGLE_USERNAME' not in os.environ:
    os.environ['KAGGLE_USERNAME'] = input('Username: ')
    os.environ['KAGGLE_KEY'] = input('Key: ')


def run_with_mod(num_extras, name=None):
    #pl.seed_everything(42)
    name = '{}_{}'.format(num_extras, '_'.join(names.get_full_name().split()))
    wandb_logger = WandbLogger(name=name,
                               project='uncertain_classification_transfer_learning')

    model = CustomModel()
    dm = modify_data_module(TBDataModule(),num_extras)
    dm.prepare_data()
    dm.setup()

    model.steps_per_epoch = len(dm.train_dataloader())

    freeze_max_epochs = 1
    unfreeze_max_epochs = 3

    model.epochs = freeze_max_epochs

    model.freeze()

    trainer = pl.Trainer(
        gpus=0,
        max_epochs=freeze_max_epochs,
        logger=wandb_logger
    )
    trainer.fit(model, dm)


    model.epochs=unfreeze_max_epochs
    model.unfreeze()

    trainer = pl.Trainer(
        gpus=0,
        max_epochs=freeze_max_epochs + unfreeze_max_epochs,
        logger=wandb_logger
    )
    trainer.current_epoch = freeze_max_epochs
    trainer.global_step = model.steps_per_epoch * freeze_max_epochs
    trainer.fit(model, dm)

    wandb.finish()

if  __name__=='__main__':
    for num_extras in (0,1,2,4,5,30,100,500,):
        run_with_mod(num_extras)

