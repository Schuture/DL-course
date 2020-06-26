from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms


class _OWN(data.Dataset):
    def __init__(self, config, is_train=True):
        
        self.mode = "train" if is_train else "test"

        if self.mode == "train":
            self.root = config.DATASET.TRAIN_ROOT
        elif self.mode == "test":
            self.root = config.DATASET.TEST_ROOT
        
        self.is_train = is_train
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.max_w = config.MODEL.IMAGE_SIZE.MAX_W

        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)
        self.transformer = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.25), \
            transforms.ToTensor(),
        ])
        self.normalization = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        txt_file = config.DATASET.JSON_FILE['train'] if is_train else config.DATASET.JSON_FILE['val']

        # convert name:indices to name:string
        self.labels = []
        with open(txt_file, 'r', encoding='utf-8') as file:
            for c in file.readlines():
                img_key = c.split(' ')[0]
                img_label = c.split(' ')[-1][:-1]
                if len(img_label) > 65: continue
                self.labels.append({img_key : img_label})
            # self.labels = [{c.split(' ')[0]: c.split(' ')[-1][:-1]} for c in file.readlines()]

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_name = list(self.labels[idx].keys())[0]
        img = Image.open(os.path.join(self.root, img_name), "r")
        img = img.convert("RGB")
        # resize
        img_h, img_w = img.size
        resize_ratio = self.inp_h / float(img_h)
        after_w = int(resize_ratio * img_w)
        if after_w == 0: after_w = 1
        img = img.resize((after_w, self.inp_h), resample=Image.BICUBIC)  # keep relative height-wdith ratio

        img = transforms.Pad(padding=(0, 0, self.max_w - after_w, 0), fill=1)(img) # padding
        # convert to CNN shape
        img = self.transformer(img) # C, H, W

        # deal with images with channels < 3
        c, h, w = img.shape
        if c < 3:
            n = 3 - c
            img_new = img
            for i in range(n):
                img_new = torch.cat((img_new, img), 0)
            img = img_new
        assert img.shape[0] == 3

        img = self.normalization(img)

        return img, idx








