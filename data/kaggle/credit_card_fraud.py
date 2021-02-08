import pytorch_lightning as pl
from pathlib import Path
import os

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
from collections import Counter

from data.kaggle import KaggleDataModule


class CreditCardFraudDataModule(KaggleDataModule):

    def __init__(self,kaggle_username=None,kaggle_key=None,top_n=28,oversample=False, undersample=False):
        super().__init__(kaggle_dataset='mlg-ulb/creditcardfraud',kaggle_username=kaggle_username,kaggle_key=kaggle_key)
        self.top_n=top_n
        self.oversample=oversample
        self.undersample=undersample
        assert not (self.oversample and self.undersample), "Only one of oversameple and undersample can be True"

    def setup(self, stage=None):
        original_data = pd.read_csv(self.data_dir/"creditcard.csv").astype('float32')

        stratSplit = StratifiedShuffleSplit(test_size=0.2, n_splits=1, random_state=42)

        stratSplit.get_n_splits(original_data, original_data.Class)
        train_index, test_index = next(iter(stratSplit.split(original_data, original_data.Class)))
        data_train = original_data.loc[train_index].reset_index().drop(['index'], 1)
        data_test = original_data.loc[test_index].reset_index().drop(['index'], 1)

        stratSplit.get_n_splits(data_train, data_train.Class)
        train_index, test_index = next(iter(stratSplit.split(data_train, data_train.Class)))
        data_val = data_train.loc[test_index].reset_index().drop(['index'], 1)
        data_train = data_train.loc[train_index].reset_index().drop(['index'], 1)

        columns = list(data_train.corr()['Class'].sort_values(key=lambda x: x.abs(), ascending=False).iloc[1:].index)
        columns = [x for x in columns if x.startswith('V')]
        relevant_columns = columns[:self.top_n]
        self.num_features=len(relevant_columns)

        X_train = np.array(data_train[relevant_columns],dtype=np.float32)
        y_train = np.array(data_train.Class, dtype=np.int)

        X_val = np.array(data_val[relevant_columns],dtype=np.float32)
        y_val = np.array(data_val.Class, dtype=np.int)

        X_test = np.array(data_test[relevant_columns],dtype=np.float32)
        y_test = np.array(data_test.Class, dtype=np.int)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        X_train = np.clip(X_train, -5, 5)
        X_val = np.clip(X_val, -5, 5)
        X_test = np.clip(X_test, -5, 5)

        if self.oversample:
            X_train,y_train = oversample(X_train,y_train)

        if self.undersample:
            X_train,y_train = undersample(X_train,y_train)

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.train_dataset=list(zip(X_train,y_train))
        self.val_dataset = list(zip(X_val, y_val))
        self.test_dataset = list(zip(X_test,y_test))

        self.counts=Counter([self.train_dataset[i][1] for i in range(len(self.train_dataset))])
        self.pos_frac = torch.tensor(1.*self.counts[1]/len(self.train_dataset))


    def get_class_weights(self,beta=1.):
        # Based on https://arxiv.org/abs/1901.05555
        n0, n1=self.counts[0],self.counts[1]
        t=n0+n1
        eff_n0 = (1-beta**n0) / (1-beta) if beta<1 else n0
        eff_n1 = (1 - beta**n1) / (1 - beta) if beta < 1 else n1
        k=t/(n0/eff_n0+n1/eff_n1)
        w=torch.tensor([eff_n0,eff_n1])
        return k/w

    def get_initial_bias_positive(self):
        return torch.log(self.pos_frac / (1 - self.pos_frac))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=2048, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=2048, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, batch_size=2048, num_workers=8)


def oversample(x,y):
    """
    Oversample the positive class

    :param x:
    :param y:
    :return:
    """
    y_pos=y[y==1]
    y_neg=y[y==0]
    x_pos=x[y==1]
    x_neg=x[y==0]

    ids = np.arange(len(y_pos))
    choices = np.random.choice(ids, len(y_neg))
    x_pos = x_pos[choices]
    y_pos = y_pos[choices]

    x=np.concatenate([x_pos,x_neg],0)
    y=np.concatenate([y_pos,y_neg],0)

    order = np.arange(len(y))
    np.random.shuffle(order)
    x = x[order]
    y = y[order]
    return x,y

def undersample(x,y):
    y_pos = y[y == 1]
    y_neg = y[y == 0]
    x_pos = x[y == 1]
    x_neg = x[y == 0]

    ids = np.arange(len(y_neg))
    choices = np.random.choice(ids, len(y_pos),replace=False)
    x_neg = x_neg[choices]
    y_neg = y_neg[choices]

    x = np.concatenate([x_pos, x_neg], 0)
    y = np.concatenate([y_pos, y_neg], 0)

    order = np.arange(len(y))
    np.random.shuffle(order)
    x = x[order]
    y = y[order]
    return x,y

if __name__=='__main__':
    dm=CreditCardFraudDataModule()
    dm.setup()
    print(len(dm.train_dataset))
    c=Counter([dm.train_dataset[i][1] for i in range(len(dm.train_dataset))])
    print(c)
    #print(next(iter(dm.train_dataloader())))
    #print(np.mean([dm.train_dataset[i] for i in range(len(dm.train_dataset))]))