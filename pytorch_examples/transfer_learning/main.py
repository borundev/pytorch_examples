import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from data.kaggle import TBDataModule
from pytorch_examples.transfer_learning.model import CustomModel

CustomModel.make_epoch_end_funcs(['Normal', 'TB'])

wandb_logger = WandbLogger(project='transfer_learning',name='bourne')


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

model.epochs = unfreeze_max_epochs
model.unfreeze()

trainer = pl.Trainer(
    gpus=0,
    max_epochs=freeze_max_epochs + unfreeze_max_epochs,
    logger=wandb_logger
)
trainer.current_epoch = freeze_max_epochs
trainer.global_step = model.steps_per_epoch * freeze_max_epochs
trainer.fit(model, dm)
