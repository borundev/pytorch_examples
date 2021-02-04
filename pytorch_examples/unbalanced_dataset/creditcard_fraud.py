import sys

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from data.kaggle.credit_card_fraud import CreditCardFraudDataModule
from pytorch_examples.transfer_learning.boilerplate import BoilerPlate
from utils.wandb.binary_classification import make_validation_epoch_end


class FinalLinear(nn.Linear):

    def __init__(self, b1, in_features):
        super().__init__(in_features,2,bias=False)
        self.b1 = nn.Parameter(torch.tensor(b1))

    def forward(self, x):
        z = super().forward(x)
        return torch.stack([z[:, 0], z[:, 1] + self.b1], 1)

class Model(BoilerPlate):

    def __init__(self,num_features,initial_bias_positive=0., class_weights=None):
        super().__init__()

        self.model = nn.Sequential(nn.Linear(num_features,16),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   FinalLinear(initial_bias_positive,16),
                                   )
        if class_weights is None:
            class_weights = [1.,1.]
        self.class_weights = torch.tensor([class_weights])

    def loss(self,inp, y):
        """
        Since pytorch doesn't have a one-hot version of cross entropy we implement it here
        :param inp:
        :param y:
        :return:
        """
        lsm = nn.LogSoftmax(1)
        yp = torch.stack([1 - y, y], 1)*self.class_weights
        return -torch.mean(torch.sum(yp * lsm(inp), 1))

    def training_step(self, batch, batch_idx):
        loss=super().training_step(batch,batch_idx)
        self.log('Bias_{}'.format('Fraud'),self.model[-1].b1)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=1e-3)



Model.validation_epoch_end=make_validation_epoch_end(pos_label='Fraud',neg_label='Honest')



if __name__ == '__main__':
    dm= CreditCardFraudDataModule()
    dm.prepare_data()
    dm.setup()
    pos_bias=dm.get_initial_bias_positive()
    class_weights = dm.get_class_weights()

    model=Model(dm.num_features,class_weights=class_weights)
    wandb_logger = WandbLogger(project='credit_card_fraud',name='class_weights')

    trainer = pl.Trainer(
        gpus=0,
        max_epochs=100,
        logger=wandb_logger
    )
    trainer.fit(model, dm)

