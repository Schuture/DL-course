import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from model.resnet import resnet18

# 构建 SummaryWriter
writer1 = SummaryWriter(comment='writer1_with_reg', filename_suffix="5_16")
writer2 = SummaryWriter(comment='writer2_without_reg', filename_suffix="5_16")

# 指定模型
activate = nn.ReLU
hidden = False

net1 = resnet18(pretrained=False, progress=True, activate=activate, hidden=hidden, num_classes=10)
net2 = resnet18(pretrained=False, progress=True, activate=activate, hidden=hidden, num_classes=10)

# 载入参数
net1.load_state_dict(torch.load('with_L2.pth', map_location=torch.device('cpu')))
net2.load_state_dict(torch.load('without_L2.pth', map_location=torch.device('cpu')))

# 记录参数并可视化
for name, param in net1.named_parameters():
    writer1.add_histogram(name + '_data', param.clone().data.numpy())
    
for name, param in net2.named_parameters():
    writer2.add_histogram(name + '_data', param.clone().data.numpy())
    
writer1.close()
writer2.close()
