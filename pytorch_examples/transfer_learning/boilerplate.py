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
        return NotImplementedError()

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

    def validation_epoch_end(self, outputs):

        ground_truths, predictions = zip(*outputs)
        predictions=torch.nn.Softmax(1)(torch.cat(predictions)).cpu()
        ground_truths=torch.cat(ground_truths).cpu()
        _,preds = torch.max(predictions,1)
        accuracy=(preds == ground_truths).type(torch.float32).mean()
        print(accuracy)

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

    def test_epoch_end(self, outputs):

        ground_truths, predictions = zip(*outputs)
        predictions=torch.nn.Softmax(1)(torch.cat(predictions)).cpu()
        ground_truths=torch.cat(ground_truths).cpu()
        _,preds = torch.max(predictions,1)
        accuracy=(preds == ground_truths).type(torch.float32).mean()
        print(accuracy)

    def configure_optimizers(self):
        NotImplementedError

    @classmethod
    def make_epoch_end_funcs(cls, labels):

        def func_train(self, outputs):
            ground_truths=[o['y'] for o in outputs]
            predictions = [o['score'] for o in outputs]
            predictions = torch.nn.Softmax(1)(torch.cat(predictions)).cpu().numpy()
            ground_truths = torch.cat(ground_truths).cpu().numpy().astype(np.int)

            self.log("train/pr", wandb.plot.pr_curve(ground_truths, predictions,
                                               labels=labels))
            self.log("train/roc", wandb.plot.roc_curve(ground_truths, predictions,
                                                 labels=labels))
            self.log('train/confusion_matrix', wandb.plot.confusion_matrix(predictions,
                                                                     ground_truths, class_names=labels))
        def func_val(self, outputs):

            ground_truths, predictions = zip(*outputs)
            predictions = torch.nn.Softmax(1)(torch.cat(predictions)).cpu().numpy()
            ground_truths = torch.cat(ground_truths).cpu().numpy().astype(np.int)

            self.log("val/pr", wandb.plot.pr_curve(ground_truths, predictions,
                                               labels=labels))
            self.log("val/roc", wandb.plot.roc_curve(ground_truths, predictions,
                                                 labels=labels))
            self.log('val/confusion_matrix', wandb.plot.confusion_matrix(predictions,
                                                                     ground_truths, class_names=labels))

        def func_test(self, outputs):
            ground_truths, predictions = zip(*outputs)
            predictions = torch.nn.Softmax(1)(torch.cat(predictions)).cpu().numpy()
            ground_truths = torch.cat(ground_truths).cpu().numpy().astype(np.int)

            self.log("test/pr", wandb.plot.pr_curve(ground_truths, predictions,
                                               labels=labels))
            self.log("test/roc", wandb.plot.roc_curve(ground_truths, predictions,
                                                 labels=labels))
            self.log('test/confusion_matrix', wandb.plot.confusion_matrix(predictions,
                                                                     ground_truths, class_names=labels))

        #cls.training_epoch_end= func_train
        cls.validation_epoch_end=func_val
        cls.test_epoch_end = func_test