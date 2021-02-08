import sklearn
import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import wandb
import torchsummary
import os, psutil
import numpy as np

from utils.wandb.binary_classification import pr_curve, roc_curve, confusion_matrix


class GAN(pl.LightningModule):

    def __init__(
            self,
            channels,
            width,
            height,
            generator,
            discriminator,
            latent_dim: int = 100,
            lr: float = 0.0002,
            b1: float = 0.5,
            b2: float = 0.999,
            batch_size: int = 64,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.data_shape = (channels, width, height)
        self.generator = generator
        self.discriminator = discriminator

        self.fixed_random_sample = None

    def print_summary(self):
        print(torchsummary.summary(self.generator, (self.hparams.latent_dim,), 1))
        print(torchsummary.summary(self.discriminator, self.data_shape, 1))

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):

        # if this is the first run make the fixed random vector
        if self.fixed_random_sample is None:
            imgs, _ = batch
            self.fixed_random_sample = torch.randn(6, self.hparams.latent_dim, device=self.device)

        # log images generatd from fixed random noise status of the fixed random noise generated images
        sample_imgs = self(self.fixed_random_sample)
        grid = torchvision.utils.make_grid(sample_imgs,padding=2, normalize=True).detach().cpu().numpy().transpose(1, 2, 0)
        self.logger.experiment.log(
            {'gen_images': [wandb.Image(grid, caption='{}:{}'.format(self.current_epoch, batch_idx))]}, commit=False)

        process = psutil.Process(os.getpid())
        self.log('memory',process.memory_info().rss/(1024**3))

        if optimizer_idx == 0:
            return self.training_step_generator(batch, batch_idx)
        elif optimizer_idx == 1:
            return self.training_step_discriminator(batch, batch_idx)

    def training_step_generator(self, batch, batch_idx):

        imgs, _ = batch
        batch_size = imgs.shape[0]

        # note z.type_as(imgs) not only type_casts but also puts on the same device
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        generated_imgs = self(z)
        generated_y_score = self.discriminator(generated_imgs)
        generated_y = torch.ones(imgs.size(0), 1, device=self.device)
        g_loss = self.adversarial_loss(generated_y_score, generated_y)

        fooling_fraction = (generated_y_score > 0.5).type(torch.float).flatten().mean()

        self.log('generator/g_loss', g_loss, prog_bar=True)
        self.log('generator/g_fooling_fraction', fooling_fraction, prog_bar=True)

        return {'loss': g_loss}


    def training_step_discriminator(self, batch, batch_idx):
        imgs, _ = batch
        batch_size = imgs.shape[0]

        # note z.type_as(imgs) not only type_casts but also puts on the same device
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        generated_imgs = self(z)
        generated_y_score = self.discriminator(generated_imgs)
        generated_y = torch.zeros(imgs.size(0), 1, device=self.device)
        generated_loss = self.adversarial_loss(generated_y_score, generated_y)

        real_y_score = self.discriminator(imgs)
        real_y = torch.ones(imgs.size(0), 1, device=self.device)
        real_loss = self.adversarial_loss(real_y_score, real_y)

        y_score = torch.cat([real_y_score, generated_y_score], 0)
        y = torch.cat([real_y, generated_y], 0)
        pred = (y_score > 0.5).type(torch.int).view(-1, 1)

        accuracy = (pred == y).type(torch.float).mean()
        loss = (real_loss + generated_loss) / 2.0

        self.log('discriminator/d_loss', loss, prog_bar=True)
        self.log('discriminator/d_accuracy', accuracy, prog_bar=True)

        return {'loss': loss,
                'y_score': y_score.detach().clone(),
                'y': y.detach().clone()
                }

    def training_epoch_end(self, outputs):
        discriminator_score = []
        discriminator_y = []

        for output in outputs[1]:
            discriminator_y.append(output['y'])
            discriminator_score.append(output['y_score'])
        discriminator_score = torch.cat(discriminator_score)
        discriminator_y = torch.cat(discriminator_y)


        y_true = discriminator_y.flatten().cpu().numpy()
        y_score = discriminator_score.flatten().cpu().numpy()
        pred = (y_score > 0.5).astype(np.int)

        neg_label,pos_label='Fake','Real'


        aps_0 = average_precision_score(1-discriminator_y, 1-discriminator_score)
        aps_1 = average_precision_score(discriminator_y, discriminator_score)
        auc_0 = roc_auc_score(1-discriminator_y, 1-discriminator_score)
        auc_1 = roc_auc_score(discriminator_y, discriminator_score)

        self.log('val/avg_precision_{}'.format(neg_label),aps_0)
        self.log('val/avg_precision_{}'.format(pos_label), aps_1)
        self.log('val/roc_auc_{}'.format(neg_label), auc_0)
        self.log('val/roc_auc_{}'.format(pos_label), auc_1)

        ax = confusion_matrix(y_true,pred,pos_label,neg_label)
        self.log('CM', wandb.Image(ax))

        ax=pr_curve(discriminator_y,discriminator_score,pos_label,neg_label)
        self.log('PR', wandb.Image(ax))

        ax=roc_curve(discriminator_y,discriminator_score,pos_label,neg_label)
        self.log('ROC', wandb.Image(ax))

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
