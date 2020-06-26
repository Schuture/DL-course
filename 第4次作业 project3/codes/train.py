''' 训练三维点云模型 '''
import sys
sys.path.append(sys.path)

import os
import random
import time
import numpy as np
import argparse
import provider

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter

from model import cls_3d
from model import get_model, get_loss
from dataset import ModelNetDataset

parser = argparse.ArgumentParser(description='PyTorch ModelNet Training')
parser.add_argument('--outf', default='./saved_curves/',
                    help='folder to output logs') # 输出结果保存路径
parser.add_argument('--outfname', default='',
                    help='filename to output logs') # 输出结果保存文件名
parser.add_argument('--seed', default=False,
                    help='random seed') # 设置随机数种子
parser.add_argument('--epoch', default=240,
                    help='max epoch of training') # 最大训练轮数
parser.add_argument('--batch', default=128,
                    help='batch size of training') # 训练时的batch size
parser.add_argument('--lr', default=0.1,
                    help='learning rate at initial') # 初始学习率
parser.add_argument('--reg', default=1e-4,
                    help='weight decay coefficient') # 权重衰减系数
parser.add_argument('--optim', default='sgd',
                    help='choose which optimizer to use') # 优化器
parser.add_argument('--scheduler', default='multisteplr',
                    help='the scheme of scheduler') # 学习率调整策略
parser.add_argument('--pointnet', action='store_true', default=False,
                    help='Whether to use Pointnet++ [default: False]')
parser.add_argument('--normal', action='store_true', default=False, 
                    help='Whether to use normal information [default: False]')
args = parser.parse_args()


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现。
    
GLOBAL_WORKER_ID = None
GLOBAL_SEED = 2 # 如果不设置随机数种子，则初始化一个种子给worker_init_fn
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


if args.seed:
    # 如果需要真的复现结果，则需要以下设置，但是CUDNN效率会降低，速度下降3成
    from torch.backends import cudnn
    cudnn.benchmark = False            # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    GLOBAL_SEED = args.seed
    set_seed(GLOBAL_SEED)


# 参数设置
MAX_EPOCH = int(args.epoch)
BATCH_SIZE = int(args.batch)
LR = float(args.lr)
L2_REG = float(args.reg)
log_interval = 1 # 多少个batch打印一次学习信息
val_interval = 1 # 多少个epoch进行一次验证
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 设置设备

# ============================ step 1/6 数据 ============================

root = '/root/CYX_Space/3d/modelnet40_ply_hdf5_2048/'
train_data_list = 'train_files.txt'
test_data_list = 'test_files.txt'
train_dataset = ModelNetDataset(root, train_data_list)
test_dataset = ModelNetDataset(root, test_data_list)
print('The number of samples in training set / testing set:')
print(len(train_dataset))
print(len(test_dataset), '\n')

trainloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
testloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# ============================ step 2/6 模型 ============================
if args.pointnet:
    net = get_model(40, normal_channel=args.normal)
else:
    net = cls_3d()
net.to(device)

# ============================ step 3/6 损失函数 ============================
if args.pointnet:
    criterion = get_loss() # 负对数似然损失
else:
    criterion = nn.CrossEntropyLoss()  # 选择损失函数

# ============================ step 4/6 优化器 ============================
# 选择优化器
if args.optim == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay = L2_REG)
elif args.optim == 'adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR, weight_decay=L2_REG)
elif args.optim == 'rmsprop':
    optimizer = optim.RMSprop(net.parameters(), lr=LR, momentum=0.9, weight_decay = L2_REG)
elif args.optim == 'adadelta':
    optimizer = optim.Adadelta(net.parameters(), lr=LR, weight_decay=L2_REG)
elif args.optim == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=L2_REG)
else:
    raise ValueError('optimizer should be sgd/adagrad/rmsprop/adadelta/adam')

# 设置学习率调整策略
if args.scheduler == 'steplr':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
elif args.scheduler == 'multisteplr':
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[135, 185], gamma=0.1)
elif args.scheduler == 'explr':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9, last_epoch=-1)
elif args.scheduler == 'coslr':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10, eta_min = 0)
else:
    raise ValueError('scheduler should be steplr/multisteplr/explr/coslr')

# ============================ step 5/6 训练 ============================
print('\nTraining start!\n')
start = time.time()

train_losscurve = list()
valid_losscurve = list()
train_acc_curve = list()
valid_acc_curve = list()

for epoch in range(MAX_EPOCH):

    loss_mean = 0.
    correct = 0.
    total = 0.

    net.train() # 切换到训练模式
    for i, data in enumerate(trainloader):

        # 读取数据并数据增强
        points, labels = data['points'], data['label'].squeeze().long() # 需要标签是Long类型
        points = points.data.numpy()
        points = provider.random_point_dropout(points) # 随机舍弃
        points[:,:,0:3] = provider.random_scale_point_cloud(points[:,:,0:3]) # 随机放缩
        points[:,:,0:3] = provider.shift_point_cloud(points[:,:,0:3]) # 随机偏移
        points = torch.Tensor(points)
        points = points.transpose(2, 1) # [Batchsize, N, C] -> [Batchsize, C, N]
        
        # forward
        points = points.to(device)
        labels = labels.to(device)
        if args.pointnet:
            outputs, trans_feat = net(points)
        else:
            outputs = net(points)

        # backward
        optimizer.zero_grad()
        if args.pointnet:
            loss = criterion(outputs, labels, trans_feat)
        else:
            loss = criterion(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().cpu().numpy()

        # 打印训练信息，记录训练损失、准确率
        loss_mean += loss.item()
        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            train_losscurve.append(round(loss_mean, 4))
            train_acc_curve.append(round(correct / total, 4))
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch+1, MAX_EPOCH, i+1, len(trainloader), loss_mean, correct / total))
            loss_mean = 0.
        
    scheduler.step()  # 更新学习率

    # validate the model
    if (epoch+1) % val_interval == 0:
        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval() # 切换到评估模式
        with torch.no_grad():
            for j, data in enumerate(testloader):
                points, labels = data['points'], data['label'].squeeze().long() # 需要标签是Long类型
                points = points.transpose(2, 1) # [Batchsize, N, C] -> [Batchsize, C, N]
                points = points.to(device)
                labels = labels.to(device)
                
                if args.pointnet:
                    outputs, trans_feat = net(points)
                    loss = criterion(outputs, labels, trans_feat)
                else:
                    outputs = net(points)
                    loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().sum().cpu().numpy()

                loss_val += loss.item()

            acc = correct_val / total_val
            valid_losscurve.append(round(loss_val/len(testloader), 4)) # 记录损失函数曲线
            valid_acc_curve.append(round(acc, 4)) # 记录准确率曲线
            print("\nValid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}\n".format(
                epoch+1, MAX_EPOCH, j+1, len(testloader), loss_val/len(testloader), correct_val / total_val))

max_train_acc = max(train_acc_curve)
max_val_acc = max(valid_acc_curve)
print('\nTraining finish, the time consumption of {} epochs is {}s\n'.format(MAX_EPOCH, round(time.time() - start)))
print('The max training accuracy is: {:.2%}\n'.format(max_train_acc))
print('The max validation accuracy is: {:.2%}\n'.format(max_val_acc))

# ============================ step 6/6 保存结果 ============================

current_time = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime())

# 保存模型参数
model_path = './saved_model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
    
torch.save(net.state_dict(), os.path.join(model_path, '3d_net_' + current_time + '_{:.2%}'.format(max_val_acc) + '.pth'))

# 保存四条曲线以便可视化
if not os.path.exists(args.outf):
    os.makedirs(args.outf)

with open(args.outf + args.outfname + current_time + '_{:.2%}'.format(max_val_acc) + '.txt', 'w') as f:
    f.write('Training loss:\n')
    f.write(str(train_losscurve))
    f.write('\n\nTraining acc:\n')
    f.write(str(train_acc_curve))
    f.write('\n\nValidation loss:\n')
    f.write(str(valid_losscurve))
    f.write('\n\nValidation acc:\n')
    f.write(str(valid_acc_curve))
    f.write('\n\nThe max training accuracy is: {:.2%}\n'.format(max_train_acc))
    f.write('\n\nThe max validation accuracy is: {:.2%}\n'.format(max_val_acc))






