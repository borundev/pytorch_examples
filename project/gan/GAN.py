import pytorch_lightning as pl
from models.gan import GAN
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule
import os

dm = MNISTDataModule(os.environ.get('PYTORCH_DATA','.'))
model = GAN(*dm.size())

from pytorch_lightning.loggers import WandbLogger


logger = WandbLogger(project='gan_mnist')

trainer = pl.Trainer(gpus=0,
                     max_epochs=5,
                     progress_bar_refresh_rate=20,
                     logger=logger,
                     limit_train_batches=120
                     )
trainer.fit(model, dm)