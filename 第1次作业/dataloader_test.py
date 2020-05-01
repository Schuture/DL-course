import time
import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root = 'D:/学习/人工智能/datasets',
                                       train = True,
                                       transform = transforms.ToTensor(),
                                       download = True)


batch_sizes = [1, 4, 64, 1024]
num_workerss = [0, 1, 4, 16]
pin_memories = [True, False]
for batch_size in batch_sizes:
    for num_workers in num_workerss:
        for pin_memory in pin_memories:
            train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size = batch_size,
                                           num_workers = num_workers,
                                           pin_memory = pin_memory,
                                           shuffle = True)
            tic = time.time()
            data_batch, label_batch = next(iter(train_loader)) # 提取一个batch的数据
            toc = time.time()
            print('batch size:', batch_size, 'num_workers:', num_workers, 
                  'pin_memory:', pin_memory, 'consume {} seconds'.format(toc - tic))










