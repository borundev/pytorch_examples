import requests
import zipfile
from io import BytesIO
from pathlib import Path
from torchvision import datasets, models, transforms
import pytorch_lightning as pl
import torchvision.datasets as dset
import torch

class CustomDataModule(pl.LightningDataModule):

    def __init__(self, path='.'):
        super().__init__()
        self.data_folder = Path(path) / 'data'
        self.data_dir = self.data_folder/'hymenoptera_data'

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def prepare_data(self):
        resp = requests.get('https://download.pytorch.org/tutorial/hymenoptera_data.zip')
        filebytes = BytesIO(resp.content)
        self.data_folder.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(filebytes) as zip_ref:
            zip_ref.extractall(self.data_folder)


    def setup(self,stage=None):
        if stage=='fit' or stage is None:
            self.train_dataset = dset.ImageFolder(self.data_dir/'train',
                                        transform=self.data_transforms['train']
                                        )
            self.val_dataset = dset.ImageFolder(self.data_dir/'val',
                                                  transform=self.data_transforms['train']
                                                  )

    def train_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=4,
                                    shuffle=True, num_workers=4)

    def val_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=4,
                                           shuffle=True, num_workers=4)


if __name__ == '__main__':
    import os
    pytorch_data=os.environ.get('PYTORCH_DATA','.')
    dm = CustomDataModule(pytorch_data)
    print(dm.data_dir)
    dm.prepare_data()
    dm.setup('fit')
    dl=dm.train_dataloader()
    l=len(dl)
    print(l)
    x,y= next(iter(dl))
    print(x.shape)
    print(y)
    dl=dm.val_dataloader()
    for x,y in dl:
        print(y)


