import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from model.resnet import resnet18

cifar_label = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 
               5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    if 'ToTensor' in str(transform_train):
        img_ = img_.detach().numpy() * 255

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]) )

    return img_


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataset = torchvision.datasets.CIFAR10(root='D:/学习/人工智能/datasets',
                                           train=False,
                                           download=False,
                                           transform=transform)
    loader = torch.utils.data.DataLoader(dataset=dataset, 
                                         batch_size=1, 
                                         shuffle=False)
    # 指定模型
    activate = nn.ReLU
    hidden = False
    
    net = resnet18(pretrained=False, progress=True, activate=activate, hidden=hidden, num_classes=10)
    
    # 载入参数
    net.load_state_dict(torch.load('ResNet18_2020_05_17_08_39_28.pth', map_location=torch.device('cpu')))
    net.eval()
    
    # 开始推理，找到分类错误的图像
    n_images = 16
    i = 0
    plt.figure(figsize=(12,12))
    for _, data in enumerate(loader):
        inputs, labels = data
        
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).squeeze().sum().cpu().numpy()

        if not correct:
            i += 1
            plt.subplot(4,4,i)
            plt.imshow(transform_invert(inputs[0], transform))
            plt.title(cifar_label[int(predicted)])
            plt.axis('off')
            
        if i == n_images:
            plt.show()
            break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    