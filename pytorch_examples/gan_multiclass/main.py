import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data.kaggle.tb_xrays import TBDataModule
from pytorch_examples.gan_multiclass.models.discriminator import DiscriminatorDCGAN_CELEBA
from pytorch_examples.gan.models.generator import GeneratorDCGAN_CELEBA
from pytorch_examples.gan_multiclass.models.gan.gan_multiclass import GAN_MultiClass

Generator = Discriminator = DataModule = None
path_pytorch_data = Path(os.environ.get('PYTORCH_DATA', ''))
path_pytorch_models = Path(os.environ.get('PYTORCH_MODELS', ''))


Generator = GeneratorDCGAN_CELEBA
Discriminator = DiscriminatorDCGAN_CELEBA
DataModule = TBDataModule

dm = DataModule()
latent_dim = 100
img_shape = (3,224,224)

generator1 = Generator(latent_dim=latent_dim, img_shape=img_shape)
generator2 = Generator(latent_dim=latent_dim, img_shape=img_shape)

discriminator = Discriminator(img_shape=img_shape)
model = GAN_MultiClass(*img_shape, latent_dim=latent_dim, generator1=generator1,generator2=generator2, discriminator=discriminator)

logger = WandbLogger(project='gan_multiclass_2')

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

checkpoint_path = path_pytorch_models / 'gan_multiclass_2/'
checkpoint_path.mkdir(parents=True, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    monitor='generator1/g_fooling_fraction',
    filepath=checkpoint_path.as_posix() + '/checkpoints-{epoch:02d}',
    save_top_k=3,
    mode='max',
    verbose=True
)

trainer = pl.Trainer(gpus=gpus,
                     max_epochs=25,
                     logger=logger,
                     #checkpoint_callback=checkpoint_callback,
                     )
trainer.fit(model, dm)
