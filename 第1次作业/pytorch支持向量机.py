import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(1)



############################## 第一步 生成数据 #################################
sample_nums = 100
mean_value = 1.7 # 一类样本均值所处的位置，例如[1.7, 1.7]
bias = 0 # 数据样本整体平移
n_data = torch.ones(sample_nums, 2)
x0 = torch.normal(mean_value * n_data, 1) + bias
y0 = -torch.ones(sample_nums) # svm需要把负例标记为-1
x1 = torch.normal(-mean_value * n_data, 1) + bias
y1 = torch.ones(sample_nums)

train_x = torch.cat((x0, x1), 0)
train_y = torch.cat((y0, y1), 0)

############################## 第二步 选择模型 #################################
class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.features = nn.Linear(2,1)
        
    def forward(self, x): # 前向传播
        x = self.features(x)
        return x
    
svm = SVM() # 实例化

############################ 第三步 选择损失函数 ###############################
def loss_fn(y_pred, y):
    zeros = torch.zeros_like(y)
    ones = torch.ones_like(y)
    return torch.sum(torch.max(zeros, ones - y*y_pred))

# 正则化方法一，直接在损失函数上添加范数项
def loss_fn_with_regularization(y_pred, y, reg = 0.0001):
    norm = torch.norm(svm.features.weight[0])
    zeros = torch.zeros_like(y)
    ones = torch.ones_like(y)
    return torch.sum(torch.max(zeros, ones - y*y_pred)) + reg * norm

############################# 第四步 选择优化器 ################################
lr = 0.01
optimizer = torch.optim.SGD(svm.parameters(), lr=lr, momentum = 0.9)
# 第二种正则化方法，在优化器上加参数衰减
optimizer = torch.optim.SGD(svm.parameters(), lr=lr, momentum = 0.9, weight_decay = 10)

############################## 第五步 模型训练 #################################
for iteration in range(20):
    
    # 前向传播
    y_pred = svm(train_x)
    
    # 计算loss，原始损失
    loss = loss_fn(y_pred.squeeze(), train_y)
    # 添加了w范数的损失
    #loss = loss_fn_with_regularization(y_pred.squeeze(), train_y, reg = 10)
    
    # 反向传播，即自动求导
    optimizer.zero_grad() # 如果梯度不清零就无法收敛
    loss.backward()
    
    # 更新参数
    optimizer.step()
    
    # 画图
    if iteration % 1 == 0:
        mask = y_pred.ge(0.0).float().squeeze() # 分类阈值为0.0，即目标为符号相同即可
        mask[mask==0] = -1 # 将分类为负的预测值变为-1，这点与逻辑回归不同
        correct = (mask == train_y).sum()
        acc = correct.item() / train_y.size(0)
        
        plt.figure(figsize = (12, 12))
        # 画数据点
        plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c = 'r', label = 'class 0')
        plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c = 'b', label = 'class 1')
        
        w0, w1 = svm.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        
        # 画线段
        plot_b = float(svm.features.bias[0].item())
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1
        
        plt.xlim(-8, 8)
        plt.ylim(-8, 8)
        plt.plot(plot_x, plot_y)
        
        plt.text(-5, 5, 'Loss={:.4f}'.format(loss.data.numpy()), \
                 fontdict = {'size': 20, 'color': 'red'})
        plt.title('Iteration: {}\nw0:{:.2f}  w1:{:.2f}  b: {:.2f}  accuracy:{:.2%}'\
                  .format(iteration, w0, w1, plot_b, acc))
        plt.legend()
        
        plt.show()
        plt.pause(0.5)
        
        if acc >= 0.99:
            break
    
    
    
    