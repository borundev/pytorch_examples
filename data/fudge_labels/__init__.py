from types import MethodType
from typing import Optional

import functools
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
        if hasattr(self.ds,'get_label'):
            y=self.ds.get_label(idx)
        else:
            _,y=self.ds[idx]
        return y

    def __getitem__(self, idx):
        x,y = self.ds[idx]
        id_extras = get_extra_indices(idx, self.num_extras, len(self))
        extras = [float(self.get_label(idx_extra)) for idx_extra in id_extras]
        z = np.concatenate([np.array([y]), extras]).astype(np.float32)
        y_modified = z.mean()
        return x, y_modified


def modify_data_module(dm,num_extras):
    dm.old_setup = dm.setup
    def setup(self,stage=None):
        self.old_setup(stage)
        self.train_dataset = GetExtraLabelsDataset(self.train_dataset, num_extras)
    dm.setup=MethodType(setup,dm)
    return dm

