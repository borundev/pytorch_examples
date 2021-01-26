from data.cats_and_dogs.cats_and_dogs import CatsAndDogsDataModule
from data.fudge_labels import  modify_data_module

dm = modify_data_module(CatsAndDogsDataModule(), 4)
dm.prepare_data()
dm.setup()
train = dm.train_dataloader()
x, y = next(iter(train))
print(y)

val = dm.val_dataloader()
x, y = next(iter(val))
print(y)

test = dm.test_dataloader()
x, y = next(iter(test))
print(y)
