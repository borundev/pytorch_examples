import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
import wandb
from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from pytorch_examples.gan.data import CelebaDataModule
from pytorch_examples.gan.models import DiscriminatorFF, DiscriminatorDCGAN, DiscriminatorDCGAN_CELEBA
from pytorch_examples.gan.models import GeneratorFF, GeneratorDCGAN, GeneratorDCGAN_CELEBA
from models.gan import GAN


data = 'CIFAR'
Generator = Discriminator = DataModule = None
path_pytorch_data = Path(os.environ.get('PYTORCH_DATA', ''))
path_pytorch_models = Path(os.environ.get('PYTORCH_MODELS', ''))

if data == 'MNIST':
    Generator = GeneratorFF
    Discriminator = DiscriminatorFF
    DataModule = MNISTDataModule
elif data == 'CIFAR':
    Generator = GeneratorDCGAN
    Discriminator = DiscriminatorDCGAN
    DataModule = CIFAR10DataModule
elif data == 'CELEBA':
    Generator = GeneratorDCGAN_CELEBA
    Discriminator = DiscriminatorDCGAN_CELEBA
    DataModule = CelebaDataModule
    path_pytorch_data = path_pytorch_data / 'celeba'

dm = DataModule(path_pytorch_data)
latent_dim = 100
img_shape = dm.size()

generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
discriminator = Discriminator(img_shape=img_shape)
model = GAN(*dm.size(), latent_dim=latent_dim, generator=generator, discriminator=discriminator)

logger = WandbLogger(project='gan_cifar_2')

dm.prepare_data()
dm.setup()
dataloader = dm.train_dataloader()
real_batch = next(iter(dataloader))

real_images = np.transpose(vutils.make_grid(real_batch[0][:6], padding=2, normalize=True).numpy(), (1, 2, 0))
logger.experiment.log({'real_sample': [wandb.Image(real_images, caption='Real Images')]})

gpus = 0
if torch.cuda.is_available():
    print('GPU Available')
    device = 'cuda:0'
    gpus = 1
else:
    print('Using CPU')
    device = 'cpu'
    gpus = 0

checkpoint_path = path_pytorch_models / 'gan_mnist/'
checkpoint_path.mkdir(parents=True, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    monitor='generator/g_fooling_fraction',
    filepath=checkpoint_path.as_posix() + '/checkpoints-{epoch:02d}',
    save_top_k=3,
    mode='max',
    verbose=True
)

trainer = pl.Trainer(gpus=gpus,
                     max_epochs=25,
                     logger=logger,
                     checkpoint_callback=checkpoint_callback,
                     limit_train_batches=10,
                     )
trainer.fit(model, dm)
