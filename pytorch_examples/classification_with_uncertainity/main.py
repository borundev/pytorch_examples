import os

import names
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger

from data.fudge_labels import modify_data_module
from data.kaggle.tb_xrays import TBDataModule
from pytorch_examples.transfer_learning.model import CustomModel
from utils.wandb.binary_classification import make_validation_epoch_end

from torch import nn

CustomModel.validation_epoch_end=make_validation_epoch_end(pos_label='Tuberculosis',neg_label='Healthy')

def loss(inp, y):
    lsm = nn.LogSoftmax(1)
    yp = torch.stack([1 - y, y], 1)
    return -torch.mean(torch.sum(yp * lsm(inp), 1))

CustomModel.loss=staticmethod(loss)

if 'KAGGLE_USERNAME' not in os.environ:
    os.environ['KAGGLE_USERNAME'] = input('Username: ')
    os.environ['KAGGLE_KEY'] = input('Key: ')


def run_with_mod(num_extras, name=None):
    #pl.seed_everything(42)
    name = '{}_{}'.format(num_extras, '_'.join(names.get_full_name().split()))
    wandb_logger = WandbLogger(name=name,
                               project='tmp')

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
    global_step = trainer.global_step
    current_epoch = trainer.current_epoch

    model.epochs=unfreeze_max_epochs
    model.unfreeze()

    trainer = pl.Trainer(
        gpus=0,
        max_epochs=freeze_max_epochs + unfreeze_max_epochs,
        logger=wandb_logger
    )
    trainer.current_epoch = current_epoch + 1
    trainer.global_step = global_step + 1
    trainer.fit(model, dm)

    wandb.finish()

if  __name__=='__main__':
    for num_extras in (5,30,):
        run_with_mod(num_extras)

