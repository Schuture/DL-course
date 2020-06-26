import os
import sys
sys.path.append('./utils')
import random
from tqdm import tqdm as tqdm
#from IPython import display

import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch

from models.vgg import VGG_A, VGG_A_BN
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#print(torch.cuda.get_device_name(3))



# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader('/root/CYX_Space/CIFAR-10 experiment/data/', train=True)
val_loader = get_cifar_loader('/root/CYX_Space/CIFAR-10 experiment/data/', train=False)
for X,y in train_loader:
    ## --------------------
    # Add code as needed
    img = np.transpose(X[0], [1,2,0])
    plt.imshow(img*0.5 + 0.5)
    plt.show()
    print('\nData loader works fine.')
    ## --------------------
    break


# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(models, optimizers, criterion, train_loader, val_loader, 
          scheduler=None, epochs_n=100, best_model_path=None):
    model1, model2, model3, model4, model5 = models
    
    model1.to(device)
    model2.to(device)
    model3.to(device)
    model4.to(device)
    model5.to(device)
    
    optimizer1, optimizer2, optimizer3, optimizer4, optimizer5 = optimizers
    
    effective_beta_smoothness = []
    
    print('Training start!')

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model1.train()
        model2.train()
        model3.train()
        model4.train()
        model5.train()

        for i, data in enumerate(train_loader):
            
            # forward
            x, y = data
            x = x.to(device)
            y = y.to(device)
            outputs1 = model1(x)
            outputs2 = model2(x)
            outputs3 = model3(x)
            outputs4 = model4(x)
            outputs5 = model5(x)
            
            # backward
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()
            optimizer5.zero_grad()
            loss1 = criterion(outputs1, y)
            loss2 = criterion(outputs2, y)
            loss3 = criterion(outputs3, y)
            loss4 = criterion(outputs4, y)
            loss5 = criterion(outputs5, y)
            loss1.backward()
            loss2.backward()
            loss3.backward()
            loss4.backward()
            loss5.backward()
            
            # update weights
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            optimizer5.step()
            
            grad1 = model1.classifier[4].weight.grad.clone()
            grad2 = model2.classifier[4].weight.grad.clone()
            grad3 = model3.classifier[4].weight.grad.clone()
            grad4 = model4.classifier[4].weight.grad.clone()
            grad5 = model5.classifier[4].weight.grad.clone()
            grads = [grad1, grad2, grad3, grad4, grad5]
            # calculate the max grad difference between grad1 ~ grad5
            max_diff = 0
            for i in range(4):
                for j in range(i+1, 5):
                    max_diff = max(max_diff, round(torch.norm(grads[i]-grads[j]).item(),4))
                    
            effective_beta_smoothness.append(max_diff)

    return effective_beta_smoothness


# Train your model
# feel free to modify
if __name__ == '__main__':
    epo = 20

    #set_random_seeds(seed_value=2020, device=device)
    lr = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
    model11 = VGG_A()
    model12 = VGG_A()
    model13 = VGG_A()
    model14 = VGG_A()
    model15 = VGG_A()
    model21 = VGG_A_BN()
    model22 = VGG_A_BN()
    model23 = VGG_A_BN()
    model24 = VGG_A_BN()
    model25 = VGG_A_BN()
    models1 = [model11, model12, model13, model14, model15]
    models2 = [model21, model22, model23, model24, model25]

    optimizer11 = torch.optim.Adam(model11.parameters(), lr = lr[0])
    optimizer12 = torch.optim.Adam(model12.parameters(), lr = lr[1])
    optimizer13 = torch.optim.Adam(model13.parameters(), lr = lr[2])
    optimizer14 = torch.optim.Adam(model14.parameters(), lr = lr[3])
    optimizer15 = torch.optim.Adam(model15.parameters(), lr = lr[4])
    optimizer21 = torch.optim.Adam(model21.parameters(), lr = lr[0])
    optimizer22 = torch.optim.Adam(model22.parameters(), lr = lr[1])
    optimizer23 = torch.optim.Adam(model23.parameters(), lr = lr[2])
    optimizer24 = torch.optim.Adam(model24.parameters(), lr = lr[3])
    optimizer25 = torch.optim.Adam(model25.parameters(), lr = lr[4])
    optimizers1 = [optimizer11, optimizer12, optimizer13, optimizer14, optimizer15]
    optimizers2 = [optimizer21, optimizer22, optimizer23, optimizer24, optimizer25]
    
    criterion = nn.CrossEntropyLoss()
    
    result = train(models1, optimizers1, criterion, train_loader, val_loader, epochs_n=epo)
    result_bn = train(models2, optimizers2, criterion, train_loader, val_loader, epochs_n=epo)
    
    with open('beta.txt', 'a') as f:
        f.write(str(result) + '\n')
        f.write(str(result_bn) + '\n')













