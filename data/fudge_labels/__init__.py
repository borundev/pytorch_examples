from typing import Optional

import functools
import pytorch_lightning as pl
from torch.utils.data import Dataset
import numpy as np

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

class GetExtraLabelsDataModule:

    def __init__(self,dm,num_extras):
        self.dm=dm
        self.num_extras = num_extras

    def __getattr__(self, item):
        return getattr(self.dm,item)


    def setup(self, stage: Optional[str] = None):
        self.dm.setup(stage)
        self.dm.train_dataset = GetExtraLabelsDataset(self.dm.train_dataset,self.num_extras)