import os
import pickle
import random
import numpy as np
from PIL import Image
import torch
import torchvision.utils as vutils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

random.seed(1)
cifar_label = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, 
               "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}


def unpickle(file): # cifar-10官方推荐，使用unpickle进行解包
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict # 返回的是一个字典，包含batch_label, lebals, filename, data四种数据


def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    if 'ToTensor' in str(transform_train):
        img_ = img_.detach().numpy() * 255

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]) )

    return img_


class CIFARDataset(Dataset):
    def __init__(self, data_dir, train = True, transform=None):
        """
        cifar-10数据集的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.DATASET, self.LABELS = self.get_img_info(data_dir, train)
        self.transform = transform

    def __getitem__(self, index): # 根据index返回数据
        img, label = self.DATASET[index], self.LABELS[index] # 此时得到的是list格式图片
        img = np.array(img, dtype = np.uint8)
        img = img.reshape((32, 32, 3), order = 'F')
        img = Image.fromarray(img).convert('RGB') # 转换为PIL图像
        img = img.rotate(270) # 将图像正放
        
        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self): # 查看样本的数量
        return len(self.LABELS)

    @staticmethod # 静态方法可以使得函数直接不通过实例化就调用，例如A.static_foo(1)
    def get_img_info(data_dir, train): # 自己定义的用来读取数据的函数
        DATASET = list()
        LABELS = list()
        
        if train: # 5个训练batch的目录
            data_dirs = [os.path.join(data_dir, 'data_batch_') + str(i) for i in range(1, 6)]
        else: # 测试batch的目录
            data_dirs = [os.path.join(data_dir, 'test_batch')]
            
        for batch_dir in data_dirs:
            batch = unpickle(batch_dir)
            data = batch[b'data'] # 10000 * 3072
            labels = batch[b'labels'] # 10000
            DATASET.extend(data)
            LABELS.extend(labels)

        return DATASET, LABELS


if __name__ == '__main__':
    data_dir = "D:/学习/人工智能/datasets/cifar-10-batches-py"
    BATCH_SIZE = 64
    
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
            ])

    test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
            ])
    
    # 构建MyDataset实例
    train_data = CIFARDataset(data_dir=data_dir, train = True, transform=train_transform)
    test_data = CIFARDataset(data_dir=data_dir, train = False, transform=test_transform)
    print('训练集大小：', len(train_data))
    print('测试集大小：', len(test_data))
    
    # 构建DataLoder，使用实例化后的数据集作为dataset
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
        
    # 提取一个batch的数据并可视化
    data_batch, label_batch = next(iter(train_loader))
    print('一个batch数据的尺寸：', data_batch.shape)
    print('一个batch标签的尺寸：', label_batch.shape)
    img_grid = vutils.make_grid(data_batch, nrow=8, normalize=False, scale_each=True)
    img_grid = transform_invert(img_grid, train_transform)
    img_grid = np.array(img_grid)
    plt.figure(figsize = (12, 12))
    plt.imshow(img_grid)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    