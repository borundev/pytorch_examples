import pytorch_lightning as pl
import torch
from torch import nn
import wandb
import numpy as np

class BoilerPlate(pl.LightningModule):


    @staticmethod
    def loss(inp, y):
        """
        Since pytorch doesn't have a one-hot version of cross entropy we implement it here
        :param inp:
        :param y:
        :return:
        """
        lsm = nn.LogSoftmax(1)
        yp = torch.stack([1 - y, y], 1)
        return -torch.mean(torch.sum(yp * lsm(inp), 1))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        score = self(x)
        _, preds = torch.max(score, 1)
        loss = self.loss(score, y)
        accuracy = (preds == y.data).type(torch.float32).mean()
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/accuracy', accuracy, prog_bar=True)
        self.log('global_step', self.global_step, prog_bar=False)
        self.log('epoch', self.current_epoch, prog_bar=False)
        return {'loss':loss, 'y':y.detach().clone().cpu(), 'score':score.detach().clone().cpu()}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        score = self(x)
        _, preds = torch.max(score, 1)
        loss = self.loss(score, y)
        accuracy=(preds == y.data).type(torch.float32).mean()
        self.log('val/loss', loss,prog_bar=True)
        self.log('val/accuracy', accuracy, prog_bar=True)
        self.log('global_step', self.global_step, prog_bar=False)
        self.log('epoch', self.current_epoch, prog_bar=False)
        return y.detach().clone().cpu(),score.detach().clone().cpu()

    def test_step(self, batch, batch_idx):
        x, y = batch
        score = self(x)
        _, preds = torch.max(score, 1)
        loss = self.loss(score, y)
        accuracy=(preds == y.data).type(torch.float32).mean()
        self.log('test/loss', loss, prog_bar=True)
        self.log('test/accuracy', accuracy, prog_bar=True)
        self.log('global_step', self.global_step, prog_bar=False)
        self.log('epoch', self.current_epoch, prog_bar=False)
        return y.detach().clone().cpu(),score.detach().clone().cpu()

    def configure_optimizers(self):
        NotImplementedError

