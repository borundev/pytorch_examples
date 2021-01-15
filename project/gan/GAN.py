import pytorch_lightning as pl
from models.gan import GAN
from project.gan.models.generator import GeneratorFF,GeneratorDCGAN
from project.gan.models.discriminator import DiscriminatorFF,DiscriminatorDCGAN

from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule
import os
import sys

data = 'CIFAR'

if data == 'MNIST':
    Generator=GeneratorFF
    Discriminator=DiscriminatorFF
    DataModule=MNISTDataModule
else:
    Generator=GeneratorDCGAN
    Discriminator=DiscriminatorDCGAN
    DataModule=CIFAR10DataModule

dm = DataModule(os.environ.get('PYTORCH_DATA','.'))
latent_dim=100
img_shape=dm.size()
print(img_shape)
generator=Generator(latent_dim=latent_dim, img_shape=img_shape)
discriminator=Discriminator(img_shape=img_shape)
model = GAN(*dm.size(),latent_dim=latent_dim, generator=generator, discriminator=discriminator)

from pytorch_lightning.loggers import WandbLogger


logger = WandbLogger(project='gan_mnist_2')

trainer = pl.Trainer(gpus=0,
                     max_epochs=25,
                     progress_bar_refresh_rate=20,
                     logger=logger,
                     )
trainer.fit(model, dm)