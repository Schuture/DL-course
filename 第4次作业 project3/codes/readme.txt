随机从数据集中读取一个样本并保存为obj文件：save_pointcloud.py

训练不同层数（3/5/7/9）的基础网络，在model/model.py中修改，注释掉某些层即可。

使用不同的数据增强方法，改动train.py中157-159行，注释掉某些增强方法即可。

使用pointnet/pointnet++：改动model/__init__.py中的注释即可二选一。

训练一个普通模型：python train.py（使用默认参数，即3层网络的最佳参数）
训练一个pointnet：nohup python -u train.py --pointnet --epoch=200 --optim=adam --scheduler=steplr --lr=0.001 --batch=24 &
训练一个pointnet++：nohup python -u train.py --pointnet --epoch=140 --optim=adam --scheduler=steplr --lr=0.001 --batch=16 &
测试一个训练好的模型：改动test.py中的模型路径，然后python test.py

可视化简单模型的准确率、损失函数曲线：训练好一个模型后，使用保存下来的log，
例如2020_06_22_14_11_00_88.09%.txt来运行 python loss_visualization.py（设置log路径）


