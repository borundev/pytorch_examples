import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from data.kaggle.tb_xrays import TBDataModule
from pytorch_examples.transfer_learning.model import CustomModel
from utils.wandb.binary_classification import make_validation_epoch_end

CustomModel.validation_epoch_end=make_validation_epoch_end(pos_label='Tuberculosis',neg_label='Healthy')

wandb_logger = WandbLogger(project='transfer_learning',name='jason')


model = CustomModel()
if 'KAGGLE_USERNAME' not in os.environ:
    os.environ['KAGGLE_USERNAME'] = input('Username: ')
    os.environ['KAGGLE_KEY'] = input('Key: ')
dm = TBDataModule()
dm.prepare_data()
dm.setup()

model.steps_per_epoch = len(dm.train_dataloader())

freeze_max_epochs = 1
unfreeze_max_epochs = 5

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

model.epochs = unfreeze_max_epochs
model.unfreeze()

trainer = pl.Trainer(
    gpus=0,
    max_epochs=freeze_max_epochs + unfreeze_max_epochs,
    logger=wandb_logger
)
trainer.current_epoch = current_epoch+1
trainer.global_step = global_step+1
trainer.fit(model, dm)
