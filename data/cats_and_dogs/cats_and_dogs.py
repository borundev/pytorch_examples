from collections import Counter

from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
import os
import requests
from sklearn.model_selection import train_test_split

from data.utils import MaintainRandomState, untar

data_url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
pytorch_data_dir=Path(os.environ.get('PYTORCH_DATA','.'))
pytorch_data_dir.mkdir(exist_ok=True)




class CatsAndDogsDataset(Dataset):

    def __init__(self, idx, all_image_files, transform=None):
        self.idx = idx
        self.len = len(idx)
        self.id_to_idx = dict(enumerate(self.idx))
        self.transform = transform
        self.all_image_files = all_image_files
        self.class_probs=None

    def __len__(self):
        return self.len

    def get_label_distribution(self):
        c=Counter()
        for idx in self.id_to_idx.values():
            filename = self.all_image_files[idx]
            y_original = int(filename.name.islower()) # 1 is dog and 0 is cat
            c[y_original]+=1
        return c

    def __getitem__(self, idx):
        filename = self.all_image_files[self.id_to_idx[idx]]
        img = Image.open(filename)
        if self.transform:
            img = self.transform(img)
        y = int(filename.name.islower()) # 1 is dog and 0 is cat

        return img, y

    def get_label(self,idx):
        filename = self.all_image_files[self.id_to_idx[idx]]
        y = int(filename.name.islower())  # 1 is dog and 0 is cat
        return y

class CatsAndDogsDataModule(pl.LightningDataModule):

    def __init__(self, force_download=False):
        super().__init__()
        self.transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]),
            }
        self.fname = pytorch_data_dir / 'cats_and_dogs.tar.gz'
        self.data_dir = pytorch_data_dir / 'cats_and_dogs'
        self.image_folder = self.data_dir / 'images'
        self.force_download_data=force_download

    def prepare_data(self):
        if self.force_download_data  or not self.data_dir.exists():
            if self.force_download_data or not self.fname.exists():
                print('Downloading {} ...'.format(self.fname))
                with open(self.fname, 'wb') as f:
                    f.write(requests.get(data_url).content)
            else:
                print('File {} already exists'.format(self.fname))
            print('Extracting ...')
            untar(self.fname, self.data_dir)
            self.unlink_bad_files()
        else:
            print('Already Downloaded and Extracted')

    def unlink_bad_files(self):
        non_jpg_removed=0
        non_rgb_removed=0
        for file in self.image_folder.glob('*'):
            if os.path.splitext(file.name)[1] not in ('.jpg','.jpeg'):
                file.unlink()
                non_jpg_removed+=1
            elif Image.open(file).mode != 'RGB':
                file.unlink()
                non_rgb_removed+=1

    def setup(self, stage=None):
        self.all_image_files = list(self.image_folder.glob('*'))
        self.num_images = len(self.all_image_files)

        with MaintainRandomState():
            train_idx, self.test_idx = train_test_split(range(self.num_images),random_state=42)

        if stage == 'fit' or stage is None:
            self.train_idx, self.val_idx = train_test_split(train_idx)
            self.train_dataset = CatsAndDogsDataset(self.train_idx, self.all_image_files,self.transforms['train'])
            self.val_dataset = CatsAndDogsDataset(self.val_idx, self.all_image_files,self.transforms['val'])
        if stage == 'test' or stage is None:
            self.test_dataset = CatsAndDogsDataset(self.test_idx, self.all_image_files,self.transforms['val'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=64, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=64, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,shuffle=False, batch_size=64, num_workers=8)

if __name__ == '__main__':
    dm=CatsAndDogsDataModule()
    dm.prepare_data()
    dm.setup()
    train=dm.train_dataloader()
    val=dm.val_dataloader()
    x,y=next(iter(train))
    print(y)
    x,y=next(iter(val))
    print(y)