import random
import tarfile
from collections import Counter

from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from pathlib import Path
import os
import requests
import torch
import numpy as np
from sklearn.model_selection import train_test_split

data_url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
pytorch_data_dir=Path(os.environ.get('PYTORCH_DATA','.'))
pytorch_data_dir.mkdir(exist_ok=True)


def get_extra_indices(idx, num_extra, size):
    return list(filter(lambda k: k != idx and k < size, [(idx // num_extra) * num_extra + i for i in range(num_extra)]))

class MaintainRandomState:
    def __enter__(self):
        self.np_state = np.random.get_state()
        self.random_state = random.getstate()
        self.torch_state = torch.get_rng_state()
        #self.torch_cuda_state = torch.cuda.get_rng_state_all()

    def __exit__(self, type, value, traceback):
        np.random.set_state(self.np_state)
        random.setstate(self.random_state)
        torch.set_rng_state(self.torch_state)
        #torch.cuda.set_rng_state_all(self.torch_cuda_state)

def untar(fname,output_dir):
    mode="r:gz" if fname.name.endswith("tar.gz") else "r:"
    with tarfile.open(fname,mode) as tar:
        tar.extractall(output_dir)

class CustomDataset(Dataset):

    def __init__(self, idx, train, all_image_files, num_extras, transform=None):
        self.idx = idx
        self.len = len(idx)
        self.id_to_idx = dict(enumerate(self.idx))
        self.transform = transform
        self.train = train
        self.all_image_files = all_image_files
        self.num_extras=num_extras
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
        y_original = float(filename.name.islower()) # 1 is dog and 0 is cat
        if self.train:
            id_extras= get_extra_indices(idx, self.num_extras, len(self))
            extras=[float(self.all_image_files[self.id_to_idx[idx_extra]].name.islower()) for idx_extra in id_extras]
            n = len(extras)+1
            z = np.concatenate([np.array([y_original]), extras])
            y = z.mean()
        else:
            y = y_original
            n = 1

        return img, torch.tensor(y_original, dtype=torch.float32), torch.tensor(y,
                                                                                dtype=torch.float32), n

class CustomDataModule(pl.LightningDataModule):

    def __init__(self, num_extras, force_download=False):
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
        self.num_extras=num_extras
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
            self.train_dataset = CustomDataset(self.train_idx,
                                               True,
                                               self.all_image_files,
                                               self.num_extras,
                                               self.transforms['train'])
            self.val_dataset = CustomDataset(self.val_idx,
                                             False,
                                             self.all_image_files,
                                             self.num_extras,
                                             self.transforms['val'])
        if stage == 'test' or stage is None:
            self.test_dataset = CustomDataset(self.test_idx,
                                              False,
                                              self.all_image_files,
                                              self.num_extras,
                                              self.transforms['val'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=4, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shufffle=False, batch_size=8, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,shuffle=False, batch_size=8, num_workers=8)

if __name__=='__main2__':
    cdm=CustomDataModule(1)
    cdm.prepare_data()
    cdm.setup()
    print(len(cdm.train_dataset)+len(cdm.val_dataset)+len(cdm.test_dataset))
    x,_,_,_=next(iter(cdm.train_dataset))

    print(cdm.train_dataset.get_label_distribution())

    print(cdm.val_dataset.get_label_distribution())
    print(cdm.test_dataset.get_label_distribution())


    print(x.shape)

if __name__ == '__main__':
    m=14
    n=5
    print([get_extra_indices(i, n, m) for i in range(m)])