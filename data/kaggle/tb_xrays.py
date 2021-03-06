from pathlib import Path
import pytorch_lightning as pl
import sklearn
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os
import numpy as np
import multiprocessing


from data.kaggle import KaggleDataModule
from data.utils import MaintainRandomState


class TBDataset(Dataset):

    def __init__(self, files,transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, 0 if fname.parent.name=='Normal' else 1

    def get_label(self,idx):
        fname = self.files[idx]
        return 0 if fname.parent.name == 'Normal' else 1


class TBDataModule(KaggleDataModule):

    def __init__(self, kaggle_username=None, kaggle_key=None, train_transform=None, val_transform=None):
        super().__init__(kaggle_dataset='tawsifurrahman/tuberculosis-tb-chest-xray-dataset',
                         kaggle_username=kaggle_username,
                         kaggle_key=kaggle_key)
        if train_transform is None:
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                #transforms.Lambda(lambda x: x.mean(0).unsqueeze(0).repeat((3, 1, 1)))
            ])
        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                #transforms.Lambda(lambda x: x.mean(0).unsqueeze(0).repeat((3, 1, 1)))
            ])
        self.transforms = {
            'train': train_transform,
            'val': val_transform,
        }


    def setup(self, stage=None):
        files = list(self.data_dir.rglob('*.jpg')) + list(self.data_dir.rglob('*.png'))
        with MaintainRandomState():
            np.random.seed(42)
            np.random.shuffle(files)
            train_files,test_files= train_test_split(files,train_size=5600,random_state=42)
            train_files,val_files = train_test_split(train_files,train_size=5040,random_state=42)

        self.train_dataset = TBDataset(train_files,self.transforms['train'])
        self.val_dataset = TBDataset(val_files,self.transforms['val'])
        self.test_dataset = TBDataset(test_files,self.transforms['val'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=64, num_workers=multiprocessing.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=64, num_workers=multiprocessing.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, batch_size=64, num_workers=multiprocessing.cpu_count())




if __name__ == '__main__':
    if 'KAGGLE_USERNAME' not in os.environ:
        os.environ['KAGGLE_USERNAME']=input('Username: ')
        os.environ['KAGGLE_KEY']= input('Key: ')
    dm=TBDataModule()
    dm.prepare_data()
    dm.setup()
    train=dm.train_dataloader()
    val=dm.val_dataloader()
    x,y=next(iter(train))
    print(y)
    x,y=next(iter(val))
    print(y)
    np.random.shuffle(dm.ds.files)
    print(dm.ds.files[:20])



