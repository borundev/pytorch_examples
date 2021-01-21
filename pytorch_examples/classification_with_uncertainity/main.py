from types import MethodType

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from data.cats_and_dogs import CatsAndDogsDataModule
from data.fudge_labels import GetExtraLabelsDataset, modify_data_module
from models.resnet import CustomModel
import wandb

def run_with_mod(num_extras, name=None, max_epochs=5):
    #pl.seed_everything(42)
    wandb_logger = WandbLogger(name=name,
                               project='uncertain_classification')
    trainer = pl.Trainer(
        gpus=0,
        max_epochs=max_epochs,
        logger=wandb_logger,
        )
    model = CustomModel()
    dm = modify_data_module(CatsAndDogsDataModule(),num_extras)

    trainer.fit(model, dm)

    wandb.finish()

if  __name__=='__main__':
    run_with_mod(2,max_epochs=1)
    #cdm = GetExtraLabelsDataModule(CatsAndDogsDataModule(), 4)
    #x,y
