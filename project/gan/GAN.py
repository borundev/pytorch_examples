import pytorch_lightning as pl
from models.gan import GAN
from project.gan.models.generator import GeneratorFF,GeneratorDCGAN, GeneratorDCGAN_CELEBA
from project.gan.models.discriminator import DiscriminatorFF,DiscriminatorDCGAN, DiscriminatorDCGAN_CELEBA
import torch
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule
from project.gan.data.celeba import CelebaDataModule
import torchvision.utils as vutils
import numpy as np
import wandb

import os
import sys
from pathlib import Path


data = 'MNIST'
Generator = Discriminator = DataModule = None
path = Path(os.environ.get('PYTORCH_DATA','.'))

if data == 'MNIST':
    Generator=GeneratorFF
    Discriminator=DiscriminatorFF
    DataModule=MNISTDataModule
elif data== 'CIFAR':
    Generator=GeneratorDCGAN
    Discriminator=DiscriminatorDCGAN
    DataModule=CIFAR10DataModule
elif data=='CELEBA':
    Generator = GeneratorDCGAN_CELEBA
    Discriminator = DiscriminatorDCGAN_CELEBA
    DataModule = CelebaDataModule
    path = path/'celeba'

dm = DataModule(path)
latent_dim=100
img_shape=dm.size()

generator=Generator(latent_dim=latent_dim, img_shape=img_shape)
discriminator=Discriminator(img_shape=img_shape)

model = GAN(*dm.size(),latent_dim=latent_dim, generator=generator, discriminator=discriminator)

from pytorch_lightning.loggers import WandbLogger


logger = WandbLogger(project='gan_mnist')

dm.prepare_data()
dm.setup()
dataloader =dm.train_dataloader()
real_batch = next(iter(dataloader))
gpus=0
if torch.cuda.is_available():
    print('GPU Available')
    device = 'cuda:0'
    gpus=1
else:
    print('Using CPU')
    device = 'cpu'
    gpus=0

real_images=np.transpose(vutils.make_grid(real_batch[0].to(device)[:6], padding=2, normalize=True).cpu().detach().numpy(),(1,2,0))
logger.experiment.log({'real_sample':[wandb.Image(real_images, caption='Real Images')]})
trainer = pl.Trainer(gpus=gpus,
                     max_epochs=5,
                     logger=logger,
                     )
trainer.fit(model, dm)