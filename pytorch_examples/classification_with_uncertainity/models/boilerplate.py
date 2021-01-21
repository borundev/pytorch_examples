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
        x, y_original, y = batch
        outputs = self(x)
        _, preds = torch.max(outputs, 1)
        loss = self.loss(outputs, y)
        loss_original = self.loss(outputs, y_original)

        self.log('train/loss_step', loss)
        self.log('train/loss', loss, on_epoch=True)

        self.log('train/loss_original_step', loss_original)
        self.log('train/loss_original', loss_original, on_epoch=True)

        self.log('train/accuracy_original_step', (preds == y_original.data).type(
            torch.float32).mean())
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
        return y,outputs

    def validation_epoch_end(self, outputs):

        ground_truths, predictions = zip(*outputs)
        predictions=torch.nn.Softmax(1)(torch.cat(predictions)).cpu().numpy()
        ground_truths=torch.cat(ground_truths).cpu().numpy().astype(np.int)

        self.log("pr", wandb.plot.pr_curve(ground_truths, predictions,
                                                labels=['Cat','Dog']))
        self.log("roc", wandb.plot.roc_curve(ground_truths, predictions,
                                                labels=['Cat','Dog']))
        self.log('confusion_matrix',wandb.plot.confusion_matrix(predictions,
                                    ground_truths,class_names=['Cat','Dog']))

    def test_step(self, batch, batch_idx):
        x, y_original, y, _ = batch
        outputs = self(x)
        _, preds = torch.max(outputs, 1)
        loss = self.loss(outputs, y)
        self.log('test/loss', loss)
        self.log('test/accuracy', (preds == y.data).type(torch.float32).mean())

    def configure_optimizers(self):
        NotImplementedError

