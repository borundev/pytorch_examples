from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pl_bolts.datamodules import MNISTDataModule
from torch import nn
import sys

from pytorch_examples.boilerplate.classification_boilerplate import BoilerPlate


class ConvBlock(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=3,
                 activation=True, bn=False, *args, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=2,
                             padding=kernel_size // 2, *args, **kwargs)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if activation:
            self.activation = nn.ReLU()

    def __len__(self):
        return 2 if hasattr(self,'bn') else 1

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        if item==0:
            return self.cnn
        elif item==1 and hasattr(self,'bn'):
            return self.bn
        else:
            raise Exception('Wrong Index')

    def forward(self, input):
        out = self.cnn(input)
        if hasattr(self, 'bn'):
            out = self.bn(out)
        if hasattr(self, 'activation'):
            out = self.activation(out)
        return out


class CustomModel(BoilerPlate):

    def __init__(self,steps_per_epoch,lr):
        super().__init__()
        self.model = nn.Sequential(ConvBlock(1, 8,kernel_size=5, activation=True),
                                   ConvBlock(8, 16, activation=True),
                                   ConvBlock(16, 32, activation=True),
                                   ConvBlock(32, 64, activation=True),
                                   ConvBlock(64,10,activation=False),
                                   # nn.AdaptiveAvgPool2d(1),
                                   #nn.Flatten(1),
                                   #nn.Linear(64, 10, bias=False)
                                   nn.Flatten()
                                   )
        self.steps_per_epoch=steps_per_epoch
        self.lr=lr

    def __getitem__(self, item):
        return self.model[item]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=self.lr,
                                                        # max_momentum=[.9,.7,.4],
                                                        # base_momentum=[.1,.2,.1],
                                                        epochs=1,
                                                        steps_per_epoch=self.steps_per_epoch,
                                                        anneal_strategy='cos',
                                                        pct_start=.25,
                                                        div_factor=25.0,
                                                        final_div_factor=100000.0/25.0,
                                                        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer],[scheduler]


class ActivationHistHook:
    num_bins = 200
    lower_lim = -10.
    upper_lim = 10.

    def __init__(self,layer_id,layer_type):
        self.stats = []
        self.weights_stats = []
        self.hook = None
        self.yticks, self.yticklabels = zip(*islice(zip(range(self.num_bins),
                                                        np.linspace(self.lower_lim, self.upper_lim, self.num_bins)), 0,
                                                    self.num_bins, 10))
        self.yticklabels = list(map(lambda x: '{:.2f}'.format(x), self.yticklabels))
        self.record=True
        self.layer_id=layer_id
        self.layer_type=layer_type

    def hist_hook(self, m, i, o):
        if self.record:
            self.stats.append(o.histc(self.num_bins, self.lower_lim, self.upper_lim).detach())
            self.weights_stats.append(m.weight.histc(self.num_bins, self.lower_lim, self.upper_lim).detach())

    def add_hook(self, m):
        self.hook = m.register_forward_hook(self.hist_hook)

    def get_hist(self):
        return torch.clamp(torch.log(torch.stack(self.stats, 1)),min=1e-15)

    def get_weights_hist(self):
        return torch.clamp(torch.log(torch.stack(self.weights_stats, 1)),min=1e-15)

    def get_yticks(self):
        return self.yticks, self.yticklabels

    def __del__(self):
        if self.hook is not None:
            self.hook.remove()


class CustomCallback(pl.Callback):

    def __init__(self, all_hooks):
        self.all_hooks=all_hooks
        self.lrs=[]
        self.momentums=[]

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        for hook in self.all_hooks:
            hook.record=False

    def on_train_epoch_start(self, trainer, pl_module):
        for hook in self.all_hooks:
            hook.record=True

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        opt=pl_module.optimizers(use_pl_optimizer=False)
        self.lrs.append(opt.param_groups[0]['lr'])
        self.momentums.append(opt.param_groups[0]['betas'][0])


if __name__ == '__main__':

    dm = MNISTDataModule()
    dm.batch_size=512
    print(dm.batch_size)
    dm.setup()
    dm.prepare_data()
    dl=dm.train_dataloader()
    steps_per_epoch=len(dl)

    model = CustomModel(steps_per_epoch,0.06)
    all_hooks=[]
    for i,k in enumerate(model.model):
        if isinstance(k,ConvBlock):
            for l in k:
                hook = ActivationHistHook(i,type(l))
                hook.add_hook(l)
                all_hooks.append(hook)

    callback=CustomCallback(all_hooks)
    trainer = pl.Trainer(
        gpus=0,
        max_epochs=1,
        # limit_train_batches=200,
        # limit_val_batches=0,
        callbacks=[callback]
    )

    trainer.fit(model, dm)
    # wandb_logger = WandbLogger(project='activation_stats')
    wandb.init(project='activation_stats')
    for hook in all_hooks:
        plt.figure(figsize=(20, 10))
        plt.imshow(hook.get_hist(), origin='lower')
        plt.yticks(*hook.get_yticks())
        wandb.log({'{}/Activation Histogram - {}'.format(hook.layer_id,hook.layer_type): wandb.Image(plt)})

        plt.figure(figsize=(20,10))
        plt.imshow(hook.get_weights_hist(), origin='lower')
        plt.yticks(*hook.get_yticks())
        wandb.log({'{}/Weights Histogram - {}'.format(hook.layer_id, hook.layer_type): wandb.Image(plt)})
    #plt.figure(figsize=(20, 10))
    #plt.plot(callback.lrs)
    #wandb.log({'Learning Rate': wandb.Image(plt)})

    #plt.figure(figsize=(20, 10))
    #plt.plot(callback.momentums)
    #wandb.log({'Momentum': wandb.Image(plt)})
