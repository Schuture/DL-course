import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

batch_size = 64
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root = 'D:/学习/人工智能/datasets',
                                       train = True,
                                       transform = transform)
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size = batch_size,
                                           shuffle = True)

data_batch, label_batch = next(iter(train_loader)) # 提取一个batch的数据

writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")
img_grid = vutils.make_grid(data_batch, nrow=8, normalize=True, scale_each=True)
writer.add_image("input img", img_grid, 0)
writer.close()