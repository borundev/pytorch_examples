import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models.resnet import CustomModel
from data import CustomDataModule
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
    cdm = CustomDataModule(num_extras=num_extras)
    trainer.fit(model, cdm)

    wandb.finish()
    return model, cdm, trainer

run_with_mod(2,max_epochs=1)

