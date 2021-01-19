import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torchvision.models import resnet18,resnet34
import torch.optim as optim
from torch.optim import lr_scheduler
import torchsummary
from pytorch_examples.transfer_learning_resnet.data.datamodule import CustomDataModule
import torch

class ResnetModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model_ft = resnet18(pretrained=True)

        num_ftrs = self.model_ft.fc.in_features
        self.fc= nn.Linear(num_ftrs, 2)
        self.model_ft.fc = self.fc

        self.criterion=nn.CrossEntropyLoss()

    def freeze_lower_layers(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model_ft(x)

    def configure_optimizers(self):
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.1)

        return [optimizer_ft],[exp_lr_scheduler]

    def training_step(self, batch,*args,**kwargs):
        x,y=batch
        y_hat = self(x)
        pred = torch.max(y_hat,1)[1]

        accuracy=torch.mean((y.flatten()==pred).type(torch.float)).detach().clone()
        loss = self.criterion(y_hat, y)
        self.log('train/loss',loss,prog_bar=True)
        self.log('train/accuracy',accuracy,prog_bar=True)
        return loss

    def validation_step(self, batch,*args,**kwargs):
        x,y=batch
        y_hat = self(x)
        pred = torch.max(y_hat,1)[1]

        accuracy=torch.mean((y.flatten()==pred).type(torch.float)).detach().clone()
        loss = self.criterion(y_hat, y)
        self.log('validation/loss',loss,prog_bar=True)
        self.log('validation/accuracy',accuracy,prog_bar=True)

if __name__=='__main__':
    import os
    pytorch_data = os.environ.get('PYTORCH_DATA', '.')
    dm=CustomDataModule(pytorch_data)
    model=ResnetModel()
    model.freeze_lower_layers()
    print(torchsummary.summary(model,(3,224,224)))
    logger = WandbLogger(project='transfer_learning')

    from pytorch_lightning.callbacks import Callback

    class MyCB(Callback):

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            opt=pl_module.optimizers(use_pl_optimizer=False)
            pl_module.log('lr',opt.param_groups[0]['lr'])


    trainer=pl.Trainer(gpus=0,max_epochs=10,logger=logger,callbacks=[MyCB()])
    trainer.fit(model,dm)
