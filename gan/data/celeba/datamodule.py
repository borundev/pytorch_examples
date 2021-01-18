import os
from pathlib import Path
import torchvision.datasets as dset
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from itertools import islice

class CelebaDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './', image_size: int =64, batch_size: int = 64, num_workers: int = 8):
        self.data_dir=Path(data_dir)
        for x in islice(self.data_dir.rglob('*'),5):
            print(x)
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.image_size=image_size

        self.dims = (3,image_size,image_size)

    def prepare_data(self):
        if not self.data_dir.exists():
            print('Download and unzip the data manually to',self.data_dir)

    def setup(self):
        self.dataset = dset.ImageFolder(root=self.data_dir,
                                   transform=transforms.Compose([
                                       transforms.Resize(self.image_size),
                                       transforms.CenterCrop(self.image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)

if __name__=='__main__':
    data_dir = Path(os.environ.get('PYTORCH_DATA', '')) / "celeba"
    dm = CelebaDataModule(data_dir)
    dm.prepare_data()
    dm.setup()
    dl = dm.train_dataloader()
    images,_=next(iter(dl))
    print(images.shape)