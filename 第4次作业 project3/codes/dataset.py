import os
import h5py
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class ModelNetDataset(Dataset):

    def __init__(self,
                 root,
                 data_list,):
        super().__init__()

        self.root = root
        self.data_list = data_list # 数据集文件列表
        self.cat = {}
        self.pts = [] # 点云
        self.labels = [] # 标签

        # We have 5 files for training and 2 files for testing
        with open(os.path.join(self.root, self.data_list)) as f:
            for file_name in f:
                print('loading: ', file_name)
                file_name = file_name.strip()
                data = h5py.File(os.path.join(self.root, file_name), 'r')
                self.pts.append(data['data'])
                self.labels.append(data['label'])
        # Combine model data from all files
        self.pts = np.vstack(self.pts)
        self.labels = np.vstack(self.labels)

    def __getitem__(self, index):

        pts = self.pts[index]
        label = self.labels[index]

        # Put the channel dimension in front for feeding into the network
        #pts = pts.transpose(1,0)

        return {
            "points": pts,
            "label": label
        }

    def __len__(self):
        return self.pts.shape[0]


if __name__ == '__main__':
    root = 'D:/学习/课程/大数据/深度学习和神经网络/作业/第4次作业 project3/modelnet40_ply_hdf5_2048'
    train_data_list = 'train_files.txt'
    test_data_list = 'test_files.txt'
    train_dataset = ModelNetDataset(root, train_data_list)
    test_dataset = ModelNetDataset(root, test_data_list)
    print(len(train_dataset))
    print(len(test_dataset))
    
    trainloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    for i, data in enumerate(trainloader, 1):
        print('The data type of data: ', type(data))
        print('The keys of data dict: ', data.keys())
        print('The shape of points: ', data['points'].shape)
        print('The data label: ', data['label'])
        print()
        if i >= 5:
            break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    