import torch
import torch.nn.functional as F
import torchvision
from project.gan.models.generator.generator_ff import Generator
from project.gan.models.discriminator.discriminator_ff import Discriminator

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

        self.fixed_random_sample=None

    def print_summary(self):
        print(torchsummary.summary(self.generator,(self.hparams.latent_dim,),1))
        print(torchsummary.summary(self.discriminator,self.data_shape,1))

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # if this is the first run make the fixed random vector
        if self.fixed_random_sample is None:
            self.fixed_random_sample = torch.randn(6, self.hparams.latent_dim)
            self.fixed_random_sample = self.fixed_random_sample.type_as(imgs)

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # generate images
        generated_imgs = self(z)

        # train generator
        if optimizer_idx == 0:

            # log sampled images
            sample_imgs = self(self.fixed_random_sample)
            grid = torchvision.utils.make_grid(sample_imgs).detach().numpy().transpose(1,2,0)

            #self.log('gen_images',[wandb.Image(grid)], on_step=True)
            self.logger.experiment.log({'gen_images': [wandb.Image(grid,caption='{}:{}'.format(self.current_epoch,batch_idx))]},commit=False)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            y = torch.ones(imgs.size(0), 1)
            y = y.type_as(imgs)

            # adversarial loss is binary cross-entropy
            y_score = self.discriminator(generated_imgs)
            fooling_fraction = (y_score > 0.5).type(torch.float).view(-1, 1).mean()
            g_loss = self.adversarial_loss(y_score, y)


            self.log('train/g_loss',g_loss,prog_bar=True)
            self.log('train/g_fooling_fraction',fooling_fraction,prog_bar=True)

            return {'loss': g_loss,
                    'optimizer_idx': optimizer_idx,
                    'y_score' : y_score,
                    'y': y
                    }

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            y = torch.cat([torch.ones(imgs.size(0), 1),torch.zeros(imgs.size(0),1)],0).type_as(imgs)

            all_images= torch.cat([imgs,generated_imgs],0)

            y_score = self.discriminator(all_images)

            loss = self.adversarial_loss(y_score,y)/2.

            pred = (y_score>0.5).type(torch.int).view(-1,1)
            accuracy = (pred==y).type(torch.float).mean()

            self.log('train/d_loss',loss,prog_bar=True)
            self.log('train/d_accuracy',accuracy,prog_bar=True)

            return {'loss': loss,
                    'optimizer_idx': optimizer_idx,
                    'y_score': y_score,
                    'y': y
                    }



    def training_epoch_end(self, outputs):
        discriminator_score=[]
        discriminator_y=[]

        for output in outputs[0]:
            if output['optimizer_idx']==1:
                print('found')
                discriminator_y.append(output['y'])
                discriminator_score.append(output['y_score'])

        for output in outputs[1]:
            if output['optimizer_idx'] == 1:
                discriminator_y.append(output['y'])
                discriminator_score.append(output['y_score'])
        discriminator_score=torch.cat(discriminator_score)
        discriminator_y=torch.cat(discriminator_y )
        discriminator_score=torch.cat([1 - discriminator_score, discriminator_score], 1)

        y_true = discriminator_y.cpu().numpy().flatten()
        y_score = discriminator_score.detach().cpu().numpy()

        self.log("train/discriminator_pr", wandb.plot.pr_curve(y_true, y_score,labels=['Fake','Real']))
        self.log("train/discriminator_roc", wandb.plot.roc_curve(y_true, y_score, labels=['Fake', 'Real']))
        self.log('train/discriminator_confusion_matrix', wandb.plot.confusion_matrix(y_score,
                                                                         y_true,class_names=['Fake','Real']))




    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

