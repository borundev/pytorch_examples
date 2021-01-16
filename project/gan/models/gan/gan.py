import torch
import torch.nn.functional as F
import torchvision
from project.gan.models.generator.generator_ff import Generator
from project.gan.models.discriminator.discriminator_ff import Discriminator
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import wandb
import torchsummary


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
            self.fixed_random_sample = torch.randn(6, self.hparams.latent_dim)
            self.fixed_random_sample = self.fixed_random_sample.type_as(imgs)

        # log images generatd from fixed random noise status of the fixed random noise generated images
        sample_imgs = self(self.fixed_random_sample)
        grid = torchvision.utils.make_grid(sample_imgs,padding=2, normalize=True).detach().cpu().numpy().transpose(1, 2, 0)
        self.logger.experiment.log(
            {'gen_images': [wandb.Image(grid, caption='{}:{}'.format(self.current_epoch, batch_idx))]}, commit=False)

        if optimizer_idx == 0:
            return self.training_step_generator(batch, batch_idx)
        elif optimizer_idx == 1:
            return self.training_step_discriminator(batch, batch_idx)

    def training_step_generator(self, batch, batch_idx):

        imgs, _ = batch
        batch_size = imgs.shape[0]

        # note z.type_as(imgs) not only type_casts but also puts on the same device
        z = torch.randn(batch_size, self.hparams.latent_dim).type_as(imgs)
        generated_imgs = self(z)
        generated_y_score = self.discriminator(generated_imgs)
        generated_y = torch.ones(imgs.size(0), 1).type_as(imgs)
        g_loss = self.adversarial_loss(generated_y_score, generated_y)

        fooling_fraction = (generated_y_score > 0.5).type(torch.float).flatten().mean()

        self.log('generator/g_loss', g_loss, prog_bar=True)
        self.log('generator/g_fooling_fraction', fooling_fraction, prog_bar=True)

        return {'loss': g_loss,
                'y_score': generated_y_score,
                'y': generated_y
                }

    def training_step_discriminator(self, batch, batch_idx):
        imgs, _ = batch
        batch_size = imgs.shape[0]

        # note z.type_as(imgs) not only type_casts but also puts on the same device
        z = torch.randn(batch_size, self.hparams.latent_dim).type_as(imgs)
        generated_imgs = self(z)
        generated_y_score = self.discriminator(generated_imgs)
        generated_y = torch.zeros(imgs.size(0), 1).type_as(imgs)
        generated_loss = self.adversarial_loss(generated_y_score, generated_y)

        real_y_score = self.discriminator(imgs)
        real_y = torch.ones(imgs.size(0), 1).type_as(imgs)
        real_loss = self.adversarial_loss(real_y_score, real_y)

        y_score = torch.cat([real_y_score, generated_y_score], 0)
        y = torch.cat([real_y, generated_y], 0)
        pred = (y_score > 0.5).type(torch.int).view(-1, 1)

        accuracy = (pred == y).type(torch.float).mean()
        loss = (real_loss + generated_loss) / 2.0

        self.log('discriminator/d_loss', loss, prog_bar=True)
        self.log('discriminator/d_accuracy', accuracy, prog_bar=True)

        return {'loss': loss,
                'y_score': y_score,
                'y': y
                }

    def training_epoch_end(self, outputs):
        discriminator_score = []
        discriminator_y = []

        for output in outputs[1]:
            discriminator_y.append(output['y'])
            discriminator_score.append(output['y_score'])
        discriminator_score = torch.cat(discriminator_score)
        discriminator_y = torch.cat(discriminator_y)
        discriminator_score = torch.cat([1 - discriminator_score, discriminator_score], 1)

        y_true = discriminator_y.cpu().numpy().flatten()
        y_score = discriminator_score.detach().cpu().numpy()

        self.log("discriminator/discriminator_pr", wandb.plot.pr_curve(y_true, y_score, labels=['Fake', 'Real']))
        self.log("discriminator/discriminator_roc", wandb.plot.roc_curve(y_true, y_score, labels=['Fake', 'Real']))
        self.log('discriminator/discriminator_confusion_matrix', wandb.plot.confusion_matrix(y_score,
                                                                                             y_true,
                                                                                             class_names=['Fake',
                                                                                                          'Real']))

        p, r, t = precision_recall_curve(1-y_true, y_score[:, 0])
        plt.plot(r,p)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        self.log('discriminator/fake_pr_curve',wandb.Image(plt,caption='Epoch: {}'.format(self.current_epoch)))
        plt.close()


    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
