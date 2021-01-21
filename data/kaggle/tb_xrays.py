import os
from pathlib import Path

import functools
from typing import Optional

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import numpy as np

class TBDataset(Dataset):

    def __init__(self, top_path):
        self.top_path = top_path
        self.files = list(self.top_path.rglob('*.jpg')) + list(self.top_path.rglob('*.png'))
        self.transform = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, 0 if fname.parent.name=='Normal' else 1


class TransformDataset(Dataset):
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def get_extra_indices(idx, num_extra, size):
    return list(filter(lambda k: k != idx and k < size, [(idx // num_extra) * num_extra + i for i in range(num_extra)]))

class GetExtraLabelsDataset(Dataset):

    def __init__(self,ds,num_extras=0):
        self.ds=ds
        self.num_extras=num_extras

    def __len__(self):
        return len(self.ds)

    @functools.lru_cache()
    def get_label(self,idx):
        _,y=self.ds[idx]
        return y

    def __getitem__(self, idx):
        x,y = self.ds[idx]
        if self.num_extras:
            id_extras = get_extra_indices(idx, self.num_extras, len(self))
            extras = [float(self.get_label(idx_extra)) for idx_extra in id_extras]
            z = np.concatenate([np.array([y]), extras])
            y_modified = z.mean()
            return x, y_modified, y
        else:
            return x,y






class TBDataModule(pl.LightningDataModule):

    def __init__(self, kaggle_username=None, kaggle_key=None, train_transform=None, val_transform=None):
        super().__init__()
        if train_transform is None:
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Lambda(lambda x: x.mean(0).unsqueeze(0).repeat((3, 1, 1)))
            ])
        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Lambda(lambda x: x.mean(0).unsqueeze(0).repeat((3, 1, 1)))
            ])
        self.transforms = {
            'train': train_transform,
            'val': val_transform,
        }
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        self.data_dir = Path(os.environ.get('PYTORCH_DATA', '.')) / 'kaggle/tuberculosis-tb-chest-xray-dataset'

    def prepare_data(self):
        if not self.data_dir.exists():
            if self.kaggle_username is not None:
                os.environ['KAGGLE_USERNAME'] = self.kaggle_username
            if self.kaggle_key is not None:
                os.environ['KAGGLE_KEY'] = self.kaggle_key
            cmd = "kaggle datasets download -p {path} --unzip tawsifurrahman/tuberculosis-tb-chest-xray-dataset".format(
                path=self.data_dir)

            os.system(cmd)

    def setup(self, stage=None):
        self.ds = TBDataset(self.data_dir)
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(self.ds,
                                                                                                [5040, 560, 1400],
                                                                                                generator=torch.Generator().manual_seed(
                                                                                                    42))
        self.train_dataset = TransformDataset(self.train_dataset,
                                              self.transforms['train'])

        self.val_dataset = TransformDataset(self.val_dataset,
                                            self.transforms['val'])
        self.test_dataset = TransformDataset(self.test_dataset,
                                             self.transforms['val'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=4, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=8, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, batch_size=8, num_workers=8)

class TBDataModuleExtraLabels(pl.LightningDataModule):

    def __init__(self,dm,num_extras):
        self.dm=dm
        self.num_extras = num_extras

    def __getattr__(self, item):
        return getattr(self.dm,item)


    def setup(self, stage: Optional[str] = None):
        self.dm.setup(stage)
        self.dm.train_dataset = GetExtraLabelsDataset(self.dm.train_dataset,self.num_extras)


if __name__ == '__main__':
    import os
    #os.environ['KAGGLE_USERNAME']=input('Username: ')
    #os.environ['KAGGLE_KEY']= input('Key: ')
    dm=TBDataModuleExtraLabels(TBDataModule(),4)
    dm.prepare_data()
    dm.setup()
    dst=dm.train_dataset
    dsv=dm.val_dataset
    x,y,y_original=dst[4]
    print(y,y_original)
    x,y=dsv[5]
    print(y)
