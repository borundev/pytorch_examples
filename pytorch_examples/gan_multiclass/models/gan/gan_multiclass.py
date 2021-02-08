import sklearn
import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import wandb
import torchsummary
import os, psutil
import numpy as np


class GAN_MultiClass(pl.LightningModule):

    def __init__(
            self,
            channels,
            width,
            height,
            generator1,
            generator2,
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
        self.generator1 = generator1
        self.generator2 = generator2
        self.discriminator = discriminator

        self.fixed_random_sample = None

        self.gen_idx=0

        self.generators = [self.generator1,self.generator2]

    def print_summary(self):
        print(torchsummary.summary(self.generator1, (self.hparams.latent_dim,), 1))
        print(torchsummary.summary(self.generator2, (self.hparams.latent_dim,), 1))
        print(torchsummary.summary(self.discriminator, self.data_shape, 1))

    def forward(self, z):
        return torch.cat([self.generator1(z[::2]),self.generator2(z[1::2])],0)

    def adversarial_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y.flatten().type(torch.long))

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

        #process = psutil.Process(os.getpid())
        #self.log('memory',process.memory_info().rss/(1024**3))

        if optimizer_idx == 0:
            return self.training_step_generator(batch, gen_idx=0)
        if optimizer_idx == 1:
            return self.training_step_generator(batch, gen_idx=1)
        if optimizer_idx == 2:
            return self.training_step_discriminator(batch)

    def training_step_generator(self, batch, gen_idx):

        imgs, _ = batch
        batch_size = imgs.shape[0]

        # note z.type_as(imgs) not only type_casts but also put s on the same device
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        generator = self.generators[gen_idx]
        generated_imgs = generator(z)

        generated_y_score = self.discriminator(generated_imgs)
        generated_y = torch.zeros(imgs.size(0), 1, device=self.device)+gen_idx
        g_loss = self.adversarial_loss(generated_y_score, generated_y)

        _, preds = torch.max(generated_y_score, 1)
        fooling_fraction = (preds == generated_y).type(torch.float32).mean()

        self.log('generator{}/loss'.format(gen_idx), g_loss, prog_bar=False)
        self.log('generator{}/fooling_fraction'.format(gen_idx), fooling_fraction, prog_bar=False)

        return {'loss': g_loss}

    def generator(self):
        gen=self.generators[self.gen_idx]
        self.gen_idx += 1
        self.gen_idx %= 2
        return gen

    def training_step_discriminator(self, batch):
        imgs, real_y = batch
        batch_size = imgs.shape[0]

        # note z.type_as(imgs) not only type_casts but also puts on the same device
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        generated_imgs = self(z)
        generated_y_score = self.discriminator(generated_imgs)
        generated_y = torch.zeros(imgs.size(0), 1, device=self.device)+2
        generated_loss = self.adversarial_loss(generated_y_score, generated_y)

        real_y_score = self.discriminator(imgs)

        real_loss = self.adversarial_loss(real_y_score, real_y)

        y_score = torch.cat([real_y_score, generated_y_score], 0)
        y = torch.cat([real_y.reshape(-1,1), generated_y], 0)
        _, pred = torch.max(y_score, 1)

        accuracy = (pred == y).type(torch.float).mean()
        loss = (real_loss + generated_loss) / 2.0

        self.log('discriminator/d_loss', loss, prog_bar=False)
        self.log('discriminator/d_accuracy', accuracy, prog_bar=False)

        return {'loss': loss,
                'y_score': y_score.detach().clone(),
                'y': y.detach().clone()
                }

    def validation_step(self, batch, batch_idx):
        imgs, real_y = batch
        batch_size = imgs.shape[0]

        # note z.type_as(imgs) not only type_casts but also puts on the same device
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        generated_imgs = self(z)
        generated_y_score = self.discriminator(generated_imgs)
        generated_y = torch.zeros(imgs.size(0), 1, device=self.device)+2
        generated_loss = self.adversarial_loss(generated_y_score, generated_y)

        real_y_score = self.discriminator(imgs)

        real_loss = self.adversarial_loss(real_y_score, real_y)

        y_score = torch.cat([real_y_score, generated_y_score], 0)
        y = torch.cat([real_y.reshape(-1,1), generated_y], 0)
        _, pred = torch.max(y_score, 1)

        accuracy = (pred == y).type(torch.float).mean()
        loss = (real_loss + generated_loss) / 2.0

        self.log('val/discriminator/d_loss', loss, prog_bar=False)
        self.log('val/discriminator/d_accuracy', accuracy, prog_bar=False)

        return {
                'y_score': y_score.detach().clone(),
                'y': y.detach().clone()
                }

    def training_epoch_end(self, outputs):
        discriminator_score = []
        discriminator_y = []

        for output in outputs[2]:
            discriminator_y.append(output['y'])
            discriminator_score.append(output['y_score'])
        discriminator_score = torch.cat(discriminator_score)
        discriminator_y = torch.cat(discriminator_y)


        y_true = discriminator_y.flatten().cpu().numpy()
        _, pred = torch.max(discriminator_score, 1)
        pred=pred.cpu().numpy()


        cm=sklearn.metrics.confusion_matrix(y_true,pred)
        fig, ax = plt.subplots(figsize=(10,10))
        ax.matshow(cm, cmap=plt.cm.Blues)
        for i in range(3):
            for j in range(3):
                c = cm[j, i]
                ax.text(i, j, str(c), va='center', ha='center', color='Red',size=20)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Healthy', 'TB', 'Fake'])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Healthy', 'TB', 'Fake'])
        plt.close()
        self.log('CM', wandb.Image(ax))

    def validation_epoch_end(self, outputs):
        discriminator_score = []
        discriminator_y = []

        for output in outputs:
            discriminator_y.append(output['y'])
            discriminator_score.append(output['y_score'])
        discriminator_score = torch.cat(discriminator_score)
        discriminator_y = torch.cat(discriminator_y)


        y_true = discriminator_y.flatten().cpu().numpy()
        _, pred = torch.max(discriminator_score, 1)
        pred=pred.cpu().numpy()

        cm=sklearn.metrics.confusion_matrix(y_true,pred,labels=[0,1,2])
        fig, ax = plt.subplots(figsize=(10,10))
        ax.matshow(cm, cmap=plt.cm.Blues)
        for i in range(3):
            for j in range(3):
                c = cm[j, i]
                ax.text(i, j, str(c), va='center', ha='center', color='Red',size=20)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Healthy', 'TB', 'Fake'])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Healthy', 'TB', 'Fake'])
        plt.close()
        self.log('CM - Val', wandb.Image(ax))


    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g1 = torch.optim.Adam(self.generator1.parameters(), lr=lr, betas=(b1, b2))
        opt_g2 = torch.optim.Adam(self.generator2.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g1, opt_g2, opt_d], []
