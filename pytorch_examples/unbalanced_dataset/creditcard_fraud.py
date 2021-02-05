import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from data.kaggle.credit_card_fraud import CreditCardFraudDataModule
from pytorch_examples.boilerplate.classification_boilerplate import BoilerPlate
from utils.wandb.binary_classification import make_validation_epoch_end
import torch.nn.functional as F
import numpy as np

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
        self.class_weights = class_weights

    def loss(self,inp, y):
        """
        Since pytorch doesn't have a one-hot version of cross entropy we implement it here
        :param inp:
        :param y:
        :return:
        """

        return F.cross_entropy(inp,y,self.class_weights)

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

    pl.seed_everything(42)
    model=Model(dm.num_features,initial_bias_positive=pos_bias, class_weights=class_weights)
    wandb_logger = WandbLogger(project='credit_card_fraud',name='initial_bias_class_weights2')

    trainer = pl.Trainer(
        gpus=0,
        max_epochs=100,
        logger=wandb_logger,
    )
    trainer.fit(model, dm)

