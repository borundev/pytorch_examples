import torchsummary
from torch import nn
import torch
import torch.optim as optim
from .boilerplate import BoilerPlate

class Net(BoilerPlate):

    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,6,3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )

        self.avgpool=nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            )

    def forward(self, x):
        x=self.features(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer

if __name__ == '__main__':
    from data import CustomDataModule

    cdm = CustomDataModule(1)
    cdm.prepare_data()
    cdm.setup()
    print(len(cdm.train_dataset) + len(cdm.val_dataset) + len(cdm.test_dataset))
    x, _, _, _ = next(iter(cdm.train_dataset))
    x=x[None,...]
    print(x.shape)

    model=Net()

    print(torchsummary.summary(model,input_size=x.shape[1:]))