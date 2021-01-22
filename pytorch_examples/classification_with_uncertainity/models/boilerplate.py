import torch
import torch.nn as nn
import pytorch_lightning as pl
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
        return NotImplementedError()

    def training_step(self, batch, batch_idx):
        x, y, y_original = batch
        outputs = self(x)
        _, preds = torch.max(outputs, 1)
        loss = self.loss(outputs, y)
        loss_original = self.loss(outputs, y_original)

        self.log('train/loss', loss)
        self.log('train/loss_original', loss_original, on_epoch=True)
        self.log('train/accuracy_original', (preds == y_original.data).type(torch.float32).mean(),
                 on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        _, preds = torch.max(outputs, 1)
        loss = self.loss(outputs, y)
        accuracy=(preds == y.data).type(torch.float32).mean()
        self.log('val/loss', loss,on_epoch=True)
        self.log('val/accuracy', accuracy,on_epoch=True)
        return y.detach().clone().cpu(),outputs.detach().clone().cpu()


    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        _, preds = torch.max(outputs, 1)
        loss = self.loss(outputs, y)
        accuracy=(preds == y.data).type(torch.float32).mean()
        self.log('test/loss', loss,on_epoch=True)
        self.log('test/accuracy', accuracy,on_epoch=True)
        return y.detach().clone().cpu(),outputs.detach().clone().cpu()

    def configure_optimizers(self):
        NotImplementedError


    @classmethod
    def make_validation_epoch_end(cls,labels):
        def func(self, outputs):
            ground_truths, predictions = zip(*outputs)
            predictions = torch.nn.Softmax(1)(torch.cat(predictions)).cpu().numpy()
            ground_truths = torch.cat(ground_truths).cpu().numpy().astype(np.int)

            self.log("pr", wandb.plot.pr_curve(ground_truths, predictions,
                                               labels=labels))
            self.log("roc", wandb.plot.roc_curve(ground_truths, predictions,
                                                 labels=labels))
            self.log('confusion_matrix', wandb.plot.confusion_matrix(predictions,
                                                                     ground_truths, class_names=labels))

        cls.validation_epoch_end=func
        cls.test_epoch_end = func
