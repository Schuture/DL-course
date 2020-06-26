''' 测试训练好的模型，得到准确率 '''
import time
import torch
from torch.utils.data import DataLoader

from model import cls_3d
from model import get_model
from dataset import ModelNetDataset


def get_accuracy(net, testloader):
    # validate the model
    correct_val = 0.
    total_val = 0.
    net.eval() # 切换到评估模式
    with torch.no_grad():
        for j, data in enumerate(testloader):
            points, labels = data['points'], data['label'].squeeze().long() # 需要标签是Long类型
            points = points.transpose(2, 1) # [Batchsize, N, C] -> [Batchsize, C, N]
            points = points.to(device)
            labels = labels.to(device)
            if pointnet:
                outputs, trans_feat = net(points)
            else:
                outputs = net(points)

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).squeeze().sum().cpu().numpy()

        acc = correct_val / total_val
    return acc


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 设置设备

    # 数据
    root = '/root/CYX_Space/3d/modelnet40_ply_hdf5_2048/'
    test_data_list = 'test_files.txt'
    test_dataset = ModelNetDataset(root, test_data_list)
    print('The number of samples in testing set:')
    print(len(test_dataset), '\n')
    testloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

    # 模型
    pointnet = False
    pth = "/root/CYX_Space/3d/codes/saved_model/3d_net_2020_06_25_15:57:38_90.44%.pth"
    state_dict = torch.load(pth)
    if pointnet:
        net = get_model(40, normal_channel=False)
    else:
        net = cls_3d()
    net.load_state_dict(state_dict)
    net.to(device)
    
    start = time.time()
    print('Testing start:')
    acc = get_accuracy(net, testloader)
    end = time.time()
    print('The testing accuracy is: {:.2%}, takes {} seconds.'.format(acc, round((end-start), 3)))




