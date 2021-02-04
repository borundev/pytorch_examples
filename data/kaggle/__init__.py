import pytorch_lightning as pl
from pathlib import Path
import os

class KaggleDataModule(pl.LightningDataModule):

    def __init__(self,kaggle_dataset, kaggle_username=None,kaggle_key=None):
        super().__init__()
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        self.kaggle_dataset=kaggle_dataset
        self.data_dir = Path(os.environ.get('PYTORCH_DATA', '.')) / 'kaggle'/ kaggle_dataset
        if not self.data_dir.exists() and self.kaggle_username is None and 'KAGGLE_USERNAME' not in os.environ:
            raise Exception('Please provide Kaggle credentials')

    def prepare_data(self):
        if not self.data_dir.exists():
            if self.kaggle_username is not None:
                os.environ['KAGGLE_USERNAME'] = self.kaggle_username
            if self.kaggle_key is not None:
                os.environ['KAGGLE_KEY'] = self.kaggle_key
            cmd = "kaggle datasets download -p {path} --unzip {kaggle_dataset}".format(
                path=self.data_dir,kaggle_dataset=self.kaggle_dataset)

            os.system(cmd)
