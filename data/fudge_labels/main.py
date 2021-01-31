from data.cats_and_dogs.cats_and_dogs import CatsAndDogsDataModule
from data.fudge_labels import  modify_data_module
from data.kaggle import TBDataModule
import numpy as np

dm = modify_data_module(TBDataModule(), 10, return_original=True)
dm.prepare_data()
dm.setup()
train = dm.train_dataloader()
x, y, y_original = next(iter(train))
print(np.corrcoef(y.numpy(),y_original.numpy()))
print(list(zip(y_original,y)))

val = dm.val_dataloader()
x, y = next(iter(val))
print(y)

test = dm.test_dataloader()
x, y = next(iter(test))
print(y)
