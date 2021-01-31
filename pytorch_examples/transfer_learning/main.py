import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from data.kaggle import TBDataModule
from pytorch_examples.transfer_learning.model import CustomModel

CustomModel.make_validation_epoch_end(['Normal', 'TB'])

wandb_logger = WandbLogger(project='transfer_learning',name='bourne')


model = CustomModel()
if 'KAGGLE_USERNAME' not in os.environ:
    os.environ['KAGGLE_USERNAME'] = input('Username: ')
    os.environ['KAGGLE_KEY'] = input('Key: ')
dm = TBDataModule()
dm.prepare_data()
dm.setup()



#path=dm.data_dir/'Dataset/'
#print(path)
#def is_tb(x):
#  return 'tb' in x.lower()
#from fastai.vision.all import *
#dls = ImageDataLoaders.from_name_func(path, get_image_files(path), valid_pct=0.2,
#        label_func=is_tb,item_tfms=Resize(224))
#sys.exit(0)

model.epochs=1
model.steps_per_epoch=len(dm.train_dataloader())
model.freeze()

freeze_max_epochs = 1
unfreeze_max_epochs = 1

trainer = pl.Trainer(
    gpus=0,
    max_epochs=freeze_max_epochs,
    logger=wandb_logger,
)
trainer.fit(model, dm)



model.unfreeze()
trainer = pl.Trainer(
    gpus=0,
    max_epochs=freeze_max_epochs+unfreeze_max_epochs,
    logger=wandb_logger,
)
trainer.current_epoch = freeze_max_epochs
trainer.global_step = model.steps_per_epoch * freeze_max_epochs
trainer.fit(model, dm)
