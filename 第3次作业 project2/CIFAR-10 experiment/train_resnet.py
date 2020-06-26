import sys
sys.path.append(sys.path)

import os
import random
import time
import numpy as np
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#from tensorboardX import SummaryWriter

from model.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./saved_curves/',
                    help='folder to output logs') # 输出结果保存路径
parser.add_argument('--boardf', default=False,
                    help='folder to output tensorboard logs') # 输出tensorboard信息保存路径
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
parser.add_argument('--reg', default=5e-4,
                    help='weight decay coefficient') # 权重衰减系数
parser.add_argument('--aug', default='2',
                    help='data augmentation type') # 数据增强的类别
parser.add_argument('--activation', default='relu',
                    help='type of activation function') # 激活函数的种类
parser.add_argument('--hidden', default=False,
                    help='whether add hidden layer and dropout or not') # 是否加隐藏层
parser.add_argument('--softmax', default=False,
                    help='whether add softmax layer or not') # 是否加softmax层
parser.add_argument('--layer', default='18',
                    help='layer of the model') # 网络层数
parser.add_argument('--optim', default='sgd',
                    help='choose which optimizer to use') # 优化器
parser.add_argument('--scheduler', default='multisteplr',
                    help='the scheme of scheduler') # 学习率调整策略
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
log_interval = 40 # 多少个batch打印一次学习信息
val_interval = 1 # 多少个epoch进行一次验证
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 设置设备

# ============================ step 1/6 数据 ============================
# 有三种选择：无增强、随机灰度加上随机遮挡、随机裁剪加随机翻转
if args.aug == '0':
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
elif args.aug == '1':
    train_transform = transforms.Compose([transforms.RandomGrayscale(p = 0.2), # 0.2的概率使用灰度图像
                                          transforms.ToTensor(),
                                          transforms.RandomErasing(), # 随机遮挡图像的一部分
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
elif args.aug == '2':
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),  #先四周填充0，再将图像随机裁剪成32*32
                                          transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
else:
    raise ValueError('aug should only be 0/1/2')

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# 构建Dataset实例
trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=train_transform)
testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=test_transform)

# 构建DataLoder，使用实例化后的数据集作为dataset，为了实验复现，为数据装载器设置随机种子函数
trainloader = torch.utils.data.DataLoader(dataset=trainset, 
                                          batch_size=BATCH_SIZE, 
                                          shuffle=True, 
                                          num_workers=2, 
                                          worker_init_fn=worker_init_fn)
testloader = torch.utils.data.DataLoader(dataset=testset, 
                                         batch_size=BATCH_SIZE, 
                                         shuffle=False, 
                                         num_workers=2, 
                                         worker_init_fn=worker_init_fn)


# ============================ step 2/6 模型 ============================
# 设置激活函数
if args.activation == 'relu':
    activate = nn.ReLU
elif args.activation == 'elu':
    activate = nn.ELU
elif args.activation == 'leakyrelu':
    activate = nn.LeakyReLU
elif args.activation == 'rrelu':
    activate = nn.RReLU
elif args.activation == 'sigmoid':
    activate = nn.Sigmoid
elif args.activation == 'tanh':
    activate = nn.Tanh
else:
    raise ValueError('activation should be relu/elu/leakyrelu/rrelu/sigmoid/tanh')
    
# 设置是否使用多一层隐藏层+dropout
hidden = int(args.hidden)


# 选择模型
if args.layer == '18':
    net = resnet18(pretrained=False, progress=True, activate=activate, hidden=hidden, num_classes=10)
elif args.layer == '34':
    net = resnet34(pretrained=False, progress=True, activate=activate, hidden=hidden, num_classes=10)
elif args.layer == '50':
    net = resnet50(pretrained=False, progress=True, activate=activate, hidden=hidden, num_classes=10)
elif args.layer == '101':
    net = resnet101(pretrained=False, progress=True, activate=activate, hidden=hidden, num_classes=10)
elif args.layer == '152':
    net = resnet152(pretrained=False, progress=True, activate=activate, hidden=hidden, num_classes=10)
else:
    raise ValueError('layer should be 18 / 34 / 50 / 101 / 152')

net.to(device)
net.initialize_weights(zero_init_residual=True)

# ============================ step 3/6 损失函数 ============================
criterion = nn.CrossEntropyLoss()  # 选择损失函数

# ============================ step 4/6 优化器 ============================
# 选择优化器
if args.optim == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay = L2_REG)
elif args.optim == 'adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR, weight_decay=L2_REG)
elif args.optim == 'rmsprop':
    optimizer = optim.RMSprop(net.parameters(), lr=0.01, momentum=0.9, weight_decay = L2_REG)
elif args.optim == 'adadelta':
    optimizer = optim.Adadelta(net.parameters(), lr=LR, weight_decay=L2_REG)
elif args.optim == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=L2_REG)
else:
    raise ValueError('optimizer should be sgd/adagrad/rmsprop/adadelta/adam')

# 设置学习率调整策略
if args.scheduler == 'steplr':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)
elif args.scheduler == 'multisteplr':
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[135, 185, 240], gamma=0.1)
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
max_acc = 0.

# 构建 SummaryWriter
if args.boardf:
    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

for epoch in range(MAX_EPOCH):

    loss_mean = 0.
    correct = 0.
    total = 0.

    net.train() # 切换到训练模式
    for i, data in enumerate(trainloader):

        # forward
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)

        # backward
        optimizer.zero_grad()
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

    # 每个epoch，记录weight, bias的grad, data
    if args.boardf:
        for name, param in net.named_parameters():
            writer.add_histogram(name + '_grad', param.grad, epoch)
            writer.add_histogram(name + '_data', param, epoch)
        
    scheduler.step()  # 更新学习率

    # validate the model
    if (epoch+1) % val_interval == 0:
        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval() # 切换到评估模式
        with torch.no_grad():
            for j, data in enumerate(testloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().sum().cpu().numpy()

                loss_val += loss.item()

            acc = correct_val / total_val
            if acc > max_acc: # 更新最大正确率
                max_acc = acc
            valid_losscurve.append(round(loss_val/len(testloader), 4)) # 记录损失函数曲线
            valid_acc_curve.append(round(correct_val / total_val, 4)) # 记录准确率曲线
            print("\nValid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}\n".format(
                epoch+1, MAX_EPOCH, j+1, len(testloader), loss_val/len(testloader), correct_val / total_val))

print('\nTraining finish, the time consumption of {} epochs is {}s\n'.format(MAX_EPOCH, round(time.time() - start)))
print('The max validation accuracy is: {:.2%}\n'.format(max_acc))

# ============================ step 6/6 保存结果 ============================

current_time = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime())

# 保存模型参数
model_path = './saved_model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
    
torch.save(net.state_dict(), os.path.join(model_path, 'ResNet' + args.layer + '_' + current_time + '.pth'))

# 保存四条曲线以便可视化
if not os.path.exists(args.outf):
    os.makedirs(args.outf)

with open(args.outf + args.outfname + current_time + '_{:.2%}'.format(max_acc) + '.txt', 'w') as f:
    f.write('Training loss:\n')
    f.write(str(train_losscurve))
    f.write('\n\nTraining acc:\n')
    f.write(str(train_acc_curve))
    f.write('\n\nValidation loss:\n')
    f.write(str(valid_losscurve))
    f.write('\n\nValidation acc:\n')
    f.write(str(valid_acc_curve))
    f.write('\n\nThe max validation accuracy is: {:.2%}\n'.format(max_acc))






