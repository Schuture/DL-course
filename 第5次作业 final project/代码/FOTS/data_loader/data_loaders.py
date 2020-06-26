import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import datasets, transforms

from base import BaseDataLoader
from .dataset import SynthTextDataset, MyDataset
from .datautils import collate_fn


class SynthTextDataLoaderFactory(BaseDataLoader): # SynthText数据集的dataloader构造类

    def __init__(self, config):
        super(SynthTextDataLoaderFactory, self).__init__(config)
        dataRoot = self.config['data_loader']['data_dir']
        self.workers = self.config['data_loader']['workers']
        ds = SynthTextDataset(dataRoot)

        self.__trainDataset, self.__valDataset = self.__train_val_split(ds)

    def train(self):
        trainLoader = torchdata.DataLoader(self.__trainDataset, num_workers = self.num_workers, batch_size = self.batch_size,
                                           shuffle = self.shuffle, collate_fn = collate_fn)
        return trainLoader

    def val(self):
        shuffle = self.config['validation']['shuffle']
        valLoader = torchdata.DataLoader(self.__valDataset, num_workers = self.num_workers, batch_size = self.batch_size,
                                         shuffle = shuffle, collate_fn = collate_fn)
        return valLoader

    def __train_val_split(self, ds):
        '''

        :param ds: dataset
        :return:
        '''
        split = self.config['validation']['validation_split']

        try:
            split = float(split)
        except:
            raise RuntimeError('Train and val splitting ratio is invalid.')

        val_len = int(split * len(ds))
        train_len = len(ds) - val_len
        train, val = torchdata.random_split(ds, [train_len, val_len])
        return train, val

    def split_validation(self):
        raise NotImplementedError


class OCRDataLoaderFactory(BaseDataLoader): # 自定义数据集的dataloader构造类，与上面的不同仅在于没有指定ds数据集

    def __init__(self, config, ds):
        super(OCRDataLoaderFactory, self).__init__(config)
        self.workers = self.config['data_loader']['workers']

        if self.config['data_loader']['use_test']:
            print('Use the whole training set and testing set! Didn\'t split dataset.')
            testRoot = self.config['data_loader']['test_dir']
            test_ds = MyDataset(testRoot)
            self.__trainDataset, self.__valDataset = ds, test_ds
        else:
            print('Split dataset to train, val parts.')
            self.__trainDataset, self.__valDataset = self.__train_val_split(ds)
        # self.__trainDataset, self.__valDataset = self.__train_val_split(ds)

    def train(self):
        trainLoader = torchdata.DataLoader(self.__trainDataset, num_workers = self.num_workers, batch_size = self.batch_size,
                                           shuffle = self.shuffle, collate_fn = collate_fn, pin_memory=True)
        return trainLoader

    def val(self):
        shuffle = self.config['validation']['shuffle']
        valLoader = torchdata.DataLoader(self.__valDataset, num_workers = self.num_workers, batch_size = self.batch_size,
                                         shuffle = shuffle, collate_fn = collate_fn, pin_memory=True)
        return valLoader

    def __train_val_split(self, ds):
        '''

        :param ds: dataset
        :return:
        '''
        split = self.config['validation']['validation_split']

        try:
            split = float(split)
        except:
            raise RuntimeError('Train and val splitting ratio is invalid.')

        val_len = int(split * len(ds))
        train_len = len(ds) - val_len
        train, val = torchdata.random_split(ds, [train_len, val_len])
        return train, val

    def split_validation(self):
        raise NotImplementedError