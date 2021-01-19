from torchvision import models
import torch.nn as nn
import torch.optim as optim

from pytorch_examples.classification_with_uncertainity.models.boilerplate import BoilerPlate


class CustomModel(BoilerPlate):

    def __init__(self):
        super().__init__()
        model_ft = models.resnet34(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False

        num_ftrs = model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Linear(num_ftrs, 2)
        self.model = model_ft

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.fc.parameters(), lr=0.001, momentum=0.9)
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

    model=CustomModel()
    #print(list(model.parameters()))
    #output=model(x)
    #print(torchsummary.summary(model,input_size=x.shape[1:]))