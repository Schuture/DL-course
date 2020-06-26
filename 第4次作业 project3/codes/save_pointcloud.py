''' 将数据集中的点云保存为.obj文件 '''

import numpy as np
from dataset import ModelNetDataset
from torch.utils.data import DataLoader

root = 'D:/学习/课程/大数据/深度学习和神经网络/作业/第4次作业 project3/modelnet40_ply_hdf5_2048'
#train_data_list = 'train_files.txt'
test_data_list = 'test_files.txt'
#train_dataset = ModelNetDataset(root, train_data_list)
test_dataset = ModelNetDataset(root, test_data_list)
#print(len(train_dataset))
print(len(test_dataset))

testloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
for i, data in enumerate(testloader, 1):
    print('The data type of data: ', type(data))
    print('The keys of data dict: ', data.keys())
    print('The shape of points: ', data['points'].shape)
    print('The data label: ', data['label'])
    print()
    if i >= 1:
        break

point_cloud = data['points']
point_cloud = np.array(point_cloud[0])
print('point cloud is {} now.'.format(point_cloud.shape))

# 保存为obj文件
with_color = True
min_x, max_x = 0, 0
min_y, max_y = 0, 0
min_z, max_z = 0, 0
with open('point_cloud.obj', 'w') as f:
    for i in range(len(point_cloud)):
        point = list(point_cloud[i])
        if point[0] < min_x: min_x = point[0]
        if point[0] > max_x: max_x = point[0]
        if point[1] < min_y: min_y = point[1]
        if point[1] > max_y: max_y = point[1]
        if point[2] < min_z: min_z = point[2]
        if point[2] > max_z: max_z = point[2]
        point = [str(p) for p in point]
        
        if with_color:
            if float(point[1]) < 0.0:
                point = ' '.join(point) + ' 1.0 0.0 0.0' # 红色点
            else:
                point = ' '.join(point) + ' 0.0 1.0 0.0' # 绿色点
        #print(point)
        f.write('v ' + point + '\n')

print('物体三个维度的范围为：',min_x, max_x, min_y, max_y, min_z, max_z)

