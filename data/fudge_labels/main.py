from data.cats_and_dogs.cats_and_dogs import CatsAndDogsDataModule
from data.fudge_labels import GetExtraLabelsDataModule

dm = GetExtraLabelsDataModule(CatsAndDogsDataModule(), 4)
dm.prepare_data()
dm.setup()
train = dm.train_dataloader()
x, y, y_original = next(iter(train))