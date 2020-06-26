import torch
import torch.nn as nn

'''
basic model for point cloud classification
'''
class cls_3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        #self.conv4 = nn.Conv1d(64, 64, 1)
        self.conv5 = nn.Conv1d(64, 128, 1)
        self.conv6 = nn.Conv1d(128, 128, 1)
        self.conv7 = nn.Conv1d(128, 128, 1)
        #self.conv8 = nn.Conv1d(128, 128, 1)
        self.conv9 = nn.Conv1d(128, 256, 1)
        self.fc = nn.Linear(256, 40)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        #x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        #x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = torch.max(x, 2, keepdim=True)[0] # 返回最后一个维度（2048）最大值，即对整个点云的特征maxpooling
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x