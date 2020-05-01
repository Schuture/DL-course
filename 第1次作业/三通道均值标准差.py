import os
import pickle
import numpy as np

def unpickle(file): # cifar-10官方推荐，使用unpickle进行解包
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict # 返回的是一个字典，包含batch_label, lebals, filename, data四种数据

base_dir = "D:/学习/人工智能/datasets/cifar-10-batches-py"

# 训练集
data = list()
for i in range(1, 6):
    data_dir = os.path.join(base_dir, 'data_batch_') + str(i)
    train_batch = unpickle(data_dir)
    data.extend(train_batch[b'data']) # 10000 * 3072
data = np.array(data)
print('训练集维度：', data.shape)
print('红色通道均值：', np.mean(data[:, :1024]))
print('绿色通道均值：', np.mean(data[:, 1024:2048]))
print('蓝色通道均值：', np.mean(data[:, 2048:]))
print('红色通道标准差：', np.std(data[:, :1024]))
print('绿色通道标准差：', np.std(data[:, 1024:2048]))
print('蓝色通道标准差：', np.std(data[:, 2048:]))

# 测试集
data_dir = os.path.join(base_dir, 'test_batch')
test_batch = unpickle(data_dir)
data = test_batch[b'data']
print('\n测试集维度：', data.shape)
print('红色通道均值：', np.mean(data[:, :1024]))
print('绿色通道均值：', np.mean(data[:, 1024:2048]))
print('蓝色通道均值：', np.mean(data[:, 2048:]))
print('红色通道标准差：', np.std(data[:, :1024]))
print('绿色通道标准差：', np.std(data[:, 1024:2048]))
print('蓝色通道标准差：', np.std(data[:, 2048:]))