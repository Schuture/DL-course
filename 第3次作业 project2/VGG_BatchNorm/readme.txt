data文件夹包含了CIFAR-10数据集的DataLoader

models文件夹包含了VGG模型的定义

utils文件夹包含了模型初始化方法的函数

VGG_Loss_Landscape.py：
训练VGG模型并记录训练过程（训练验证准确率曲线、两步训练梯度差变化曲线、损失值变化）

visualize_Acc_Landscape.py, visualize_Loss_Landscape.py:
可视化保存下来的准确率曲线以及损失曲线

visualize_grad_predictiveness.py:
可视化梯度预测性（相邻两步梯度差的大小）

compare_beta_smoothness.py:
分别训练是否包含BN层的5组模型，并比较每一步的梯度差异幅度，将结果保存在beta.txt

visualize_beta_smoothness.py:
使用beta.txt来可视化beta smoothness