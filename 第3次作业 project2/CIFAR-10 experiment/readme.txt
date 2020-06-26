model文件夹包含了resnet的定义以及改动。

train_resnet.py：
模型的训练，更改数据集路径后直接运行即可，可以具体查看代码来进行命令行参数设置。

loss_visualization.py：
模型损失函数的可视化，需要使用模型训练后保存的准确率、损失函数数据。

reg_visualization.py：
需要使用保存后的是/否包含正则化的两个网络参数来可视化神经网络第一层卷积的参数分布。

CIFAR-10_visualization.py：
使用CIFAR-10数据集来进行图像的提取，并且可视化。

weight_fmap_visualization.py：
可视化训练好的模型的第一层卷积核以及这层输出的特征图。

case_study.py：
使用训练好的模型来进行错误分类样本的可视化。