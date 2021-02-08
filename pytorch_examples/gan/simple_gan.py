import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.latent_dim = 3
        self.model=nn.Linear(self.latent_dim,28*28)

    def forward(self,z):
        z=self.model(z)
        return z.view(-1,1,28,28)

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(28*28,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,z):
        z=z.view(-1,784)
        return self.sigmoid(self.model(z))


class Gan(pl.LightningModule):

    def __init__(self, gen, disc,
                 latent_dim: int = 3,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = 64,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.gen = gen
        self.disc = disc

    def forward(self, x):
        return self.gen(x)

    def training_step(self, batch, batch_idx, optimizer_idx):

        if optimizer_idx == 0:
            r = self.training_step_generator(batch)
        elif optimizer_idx == 1:
            r = self.training_step_discriminator(batch)
        return r

    def training_step_generator(self, batch):

        imgs, _ = batch
        batch_size = imgs.shape[0]

        # note z.type_as(imgs) not only type_casts but also puts on the same device
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        generated_imgs = self(z)
        generated_y_score = self.disc(generated_imgs)
        generated_y = torch.ones(imgs.size(0), 1, device=self.device)
        g_loss = self.adversarial_loss(generated_y_score, generated_y)

        self.log('generator/g_loss', g_loss, prog_bar=True)

        return {'loss': g_loss}

    def training_step_discriminator(self, batch):

        imgs, _ = batch
        batch_size = imgs.shape[0]

        # note z.type_as(imgs) not only type_casts but also puts on the same device
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        generated_imgs = self(z)
        generated_y_score = self.disc(generated_imgs)
        generated_y = torch.zeros(imgs.size(0), 1, device=self.device)
        g_loss = self.adversarial_loss(generated_y_score, generated_y)

        real_y_score = self.disc(imgs)
        real_y = torch.ones(imgs.size(0), 1, device=self.device)
        r_loss = self.adversarial_loss(real_y_score, real_y)

        d_loss = (r_loss + g_loss) / 2.0

        self.log('discriminator/d_loss', d_loss, prog_bar=True)

        return {'loss': d_loss}

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.gen.parameters())
        opt_g.name = 'G'
        opt_d = torch.optim.Adam(self.disc.parameters())
        opt_d.name = 'D'
        return [opt_g, opt_d], []


class NewCallback(pl.callbacks.Callback):

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        try:
            print('train_batch_start_generator', torch.tensor([torch.sum(p.grad != 0) for p in gen.parameters()]).sum())
            print('train_batch_start_discriminator',
                  torch.tensor([torch.sum(p.grad != 0) for p in disc.parameters()]).sum())
            print('-'*20)
        except:
            pass

    def on_after_backward(self, trainer, pl_module):
        try:
            print('after_backward_generator', torch.tensor([torch.sum(p.grad != 0) for p in gen.parameters()]).sum())
            print('after_backward_discriminator',
                  torch.tensor([torch.sum(p.grad != 0) for p in disc.parameters()]).sum())
            print('-' * 20)
        except:
            pass

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        try:
            print('before_zero_grad_generator', optimizer.name,
                  torch.tensor([torch.sum(p.grad != 0) for p in gen.parameters()]).sum())
            print('before_zero_grad_discriminator', optimizer.name,
                  torch.tensor([torch.sum(p.grad != 0) for p in disc.parameters()]).sum())
            print('-' * 20)
        except:
            pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        try:
            print('train_batch_end_generator', torch.tensor([torch.sum(p.grad != 0) for p in gen.parameters()]).sum())
            print('train_batch_end_discriminator',
                  torch.tensor([torch.sum(p.grad != 0) for p in disc.parameters()]).sum())
        except:
            pass
        print('*' * 100)

def gen_params():
    r=None
    try:
        r=torch.tensor([torch.sum(p.grad != 0) for p in gen.parameters()]).sum()
    except TypeError as e:
        pass
    return r

def disc_params():
    r=None
    try:
        r=torch.tensor([torch.sum(p.grad != 0) for p in disc.parameters()]).sum()
    except TypeError as e:
        pass
    return r

if __name__ == '__main__':
    gen=Generator()
    disc=Discriminator()
    z=torch.rand(10,100)

    imgs=torch.rand((10,1,28,28))
    labels=torch.tensor(np.random.choice([0,1],10))

    data=DataLoader(list(zip(imgs,labels)),batch_size=2)
    gan=Gan(gen,disc)
    trainer = pl.Trainer(gpus=0,
                         max_epochs=5,
                         limit_train_batches=2,
                         limit_val_batches=0,
                         progress_bar_refresh_rate=50,
                         callbacks=[NewCallback()]
                         )
    trainer.fit(gan,data)
