import os
import numpy as np
import torch
import random
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tools.my_dataset import RMBDataset
from tools.common_tools import transform_invert



def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed(1)  # 设置随机种子

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
rmb_label = {"1": 0, "100": 1}


# ============================ step 1/5 数据 ============================
split_dir = os.path.join("data", "rmb_split")
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    
    # 1 Crop
#     transforms.RandomCrop(size = 160, padding = 4, pad_if_needed = False),
    
    # 2 Resize
#     transforms.Resize((100, 100)),
    
    # 3 Flip
#    transforms.RandomHorizontalFlip(p = 1),
#    transforms.RandomVerticalFlip(p = 1),

    # 1 Pad
#     transforms.Pad(padding=32, fill=(0, 0, 0), padding_mode='constant'), # 四周32
#     transforms.Pad(padding=(8, 64), fill=(255, 0, 0), padding_mode='constant'), # 左右8，上下64
#     transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='constant'), # 左8，上16，右32，下64
#     transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='symmetric'), # 左8，上16，右32，下64，对称填补

    # 2 ColorJitter
#     transforms.ColorJitter(brightness=0.5), # 0.5 - 1.5亮度随机
#     transforms.ColorJitter(contrast=0.5), # 0.5 - 1.5对比度随机
#     transforms.ColorJitter(saturation=0.5), # 0.5 - 1.5饱和度随机
#     transforms.ColorJitter(hue=0.3), # 0.7 - 1.3色相随机

    # 3 Grayscale
#     transforms.Grayscale(num_output_channels=3), # 从三通道转换为灰度

    # 4 Affine
#     transforms.RandomAffine(degrees=30), # 旋转角度30以内随机
#     transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), fillcolor=(255, 0, 0)), # 宽、高20%以内随机平移，填充红色
#     transforms.RandomAffine(degrees=0, scale=(0.7, 0.7)), # 宽高缩小到0.7倍
#     transforms.RandomAffine(degrees=0, shear=(0, 0, 45, 45)), # 四个角度都可以在0-45度之间错切
#     transforms.RandomAffine(degrees=0, shear=90, fillcolor=(255, 0, 0)), # 一个角度在0-90度之间错切

    # 5 Erasing
#     transforms.ToTensor(), # RandomErasing函数接受的是一个Tensor
#     transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254/255, 0, 0)), # 遮挡0.02-0.33比例面积
#     transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='1234'), # 随机填充像素

    # 1 RandomChoice
#     transforms.RandomChoice([transforms.RandomVerticalFlip(p=1), transforms.RandomHorizontalFlip(p=1)]),

    # 2 RandomApply
#     transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=45, fillcolor=(255, 0, 0)),
#                             transforms.Grayscale(num_output_channels=3)], p=0.5),
    # 3 RandomOrder
#     transforms.RandomOrder([transforms.RandomRotation(15),
#                             transforms.Pad(padding=32),
#                             transforms.RandomAffine(degrees=0, translate=(0.01, 0.1), scale=(0.9, 1.1))]),

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 构建MyDataset实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)


# ============================ step 5/5 训练 ============================
for epoch in range(MAX_EPOCH):
    for i, data in enumerate(train_loader):

        inputs, labels = data   # B C H W

        img_tensor = inputs[0, ...]     # C H W
        img = transform_invert(img_tensor, train_transform)
#        plt.figure(figsize = (12,12))
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()






