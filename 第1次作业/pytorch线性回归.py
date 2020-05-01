import torch
import matplotlib.pyplot as plt
torch.manual_seed(1)


# 第一步 生成数据
x = torch.rand(20,1) * 10
y = 3 * x + (5 + torch.randn(20,1))

lr = 1e-5
w = torch.randn((1), requires_grad = True)
b = torch.randn((1), requires_grad = True)

for iteration in range(1000):
    
    # 数据通过模型的前向传播
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)
    
    # 损失函数，计算 MSE loss，有0.5是为了方便求导
    loss = (0.5 * (y-y_pred) ** 2).mean()
    
    # 反向传播，即自动求导
    loss.backward()
    
    # 优化，手动更新参数
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)
    
    # 绘图
    if iteration % 20 == 0:
        plt.figure(figsize = (12,12))
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size':20, 'color': 'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 32)
        plt.title('Iteration: {}\nw: {} b:{}'.format(iteration, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)
        
        if loss.data.numpy() < 2:
            break