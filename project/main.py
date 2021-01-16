import torch
from torch.optim import Adam

print(torch.cuda.is_available())
import wandb


import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch import nn
import numpy as np

class CustomDataSet:

    def __init__(self):
        self.x=torch.tensor(np.linspace(0,2*np.pi,100))
        self.y=torch.tensor(np.sin(self.x))
        self.x=self.x/(2*np.pi)

    def __len__(self):
        return 100

    def __getitem__(self, item):
        return self.x[item],self.y[item]

class CustomDataModule(pl.LightningDataModule):

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(CustomDataSet(),batch_size=4)

class CustomModule(pl.LightningModule):

    def __init__(self):
        self.model=nn.Sequential(
            nn.Linear(1,10),
            nn.ReLU(),
            nn.Linear(10,2),
            nn.ReLU(),
            nn.Linear(2,1),
            nn.Tanh()
        )

    def forward(self,x):
        return self.model(x)

    def criterion(self,y,y_pred):
        return nn.MSELoss()(y,y_pred)

    def training_step(self, batch, batch_idx):
        x,y= batch
        y_pred = self(x)
        loss=self.criterion(y,y_pred)
        self.log('train/loss',loss)
        return {'loss':loss}

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=1e-3)
        return opt

model = CustomModule()
dm = CustomDataModule()
trainer=pl.Trainer(model,dm)
