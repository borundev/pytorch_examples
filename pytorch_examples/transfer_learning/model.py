import torch
from torchvision.models import resnet34, resnet18
from torch import nn, optim
import numpy as np

from pytorch_examples.transfer_learning.boilerplate import BoilerPlate

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

def params(*args):
    return [{'params':list(a.parameters())} for a in args]


class CustomModel(BoilerPlate):

    @staticmethod
    def create_new_model():
        model_resnet = resnet34(pretrained=True)

        bottom = nn.Sequential(model_resnet.conv1,
                               model_resnet.bn1,
                               model_resnet.relu,
                               model_resnet.maxpool)

        num_ftrs = model_resnet.fc.in_features
        top = nn.Sequential(
            AdaptiveConcatPool2d(),
            nn.Flatten(1),
            nn.BatchNorm1d(num_ftrs * 2),
            nn.Dropout(),
            nn.Linear(num_ftrs * 2, num_ftrs, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(),
            nn.Linear(num_ftrs, 2, bias=False)
        )

        model = nn.Sequential(bottom,
                              model_resnet.layer1,
                              model_resnet.layer2,
                              model_resnet.layer3,
                              model_resnet.layer4,
                              top,
                              )
        return model

    def __init__(self, lr=.002):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.model = CustomModel.create_new_model()

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for n, l in list(self.model.named_children())[:-1]:
            getattr(self.model, n).requires_grad_(False)
        self.configure_optimizers = self.make_transfer_optimizer

    def unfreeze(self):
        for n, l in list(self.model.named_children())[:-1]:
            getattr(self.model, n).requires_grad_(True)
        self.configure_optimizers = self.make_optimizer

    @staticmethod
    def make_lrs(lr,n,div=100):
        if n==1:
            return [lr]
        else:
            return list(np.exp(np.linspace(np.log(lr/div),np.log(lr),n)))

    def set_epochs(self,epochs):
        self.epochs=epochs

    def set_steps_per_epoch(self,steps_per_epoch):
        self.steps_per_epoch=steps_per_epoch


    def make_transfer_optimizer(self):
        optimizer = optim.AdamW(params(self.model[:3], self.model[3:5], self.model[5]),eps=1e-5,betas=(0.9,0.99))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=self.make_lrs(self.lr,3),
                                                        # max_momentum=[.9,.7,.4],
                                                        # base_momentum=[.1,.2,.1],
                                                        epochs=self.epochs,
                                                        steps_per_epoch=self.steps_per_epoch,
                                                        anneal_strategy='cos',
                                                        pct_start=.99,
                                                        div_factor=25.0,
                                                        final_div_factor=100000.0/25.0,
                                                        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer],[scheduler]

    def make_optimizer(self):
        optimizer = optim.AdamW(params(self.model[:3], self.model[3:5], self.model[5]),eps=1e-5,betas=(0.9,0.99))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=self.make_lrs(self.lr/2,3),
                                                        # max_momentum=[.9,.7,.4],
                                                        # base_momentum=[.1,.2,.1],
                                                        epochs=self.epochs,
                                                        steps_per_epoch=self.steps_per_epoch,
                                                        anneal_strategy='cos',
                                                        pct_start=.3,
                                                        div_factor=5.0,
                                                        final_div_factor=100000.0/5.0,
                                                        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step', # or 'epoch'
            'frequency': 1
        }
        return [optimizer],[scheduler]


