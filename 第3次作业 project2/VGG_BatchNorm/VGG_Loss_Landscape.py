import os
import sys
sys.path.append('./utils')
import random
import time
#from tqdm import tqdm as tqdm
#from IPython import display

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch

from models.vgg import VGG19, VGG19_BN, VGG_A, VGG_A_BN, VGG_A_Dropout
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
def train(model, optimizer, criterion, train_loader, val_loader, 
          scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [0]                            # record training loss
    train_accuracy_curve = []                       # record training acc
    val_accuracy_curve = [np.nan] * epochs_n        # record validation acc
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader) # number of batches in each epoch
    interval = batches_n // 10
    grads_diff = [] # l2 distance of grad of ith step and i-1 th step
    last_grad = 0
    index = 0 # index of loss, acc record
    
    print('Training start!')
    start = time.time()
    for epoch in range(epochs_n):
        if scheduler is not None:
            scheduler.step()
        model.train()

        learning_correct = 0        # correct samples for now
        total = 0                   # total samples for now

        for i, data in enumerate(train_loader):
            
            # forward
            x, y = data
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            
            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            
            # update weights
            optimizer.step()
            
            learning_curve[index] += loss.item()
            
            grad = model.classifier[4].weight.grad.clone()
            grads_diff.append(round(torch.norm(grad-last_grad).item(), 3))
            last_grad = grad
            
            # calculate correct samples and training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            learning_correct += (predicted == y).squeeze().sum().cpu().numpy()
            
            # record 10 loss and grad and accuracy 10 times every epoch
            if (i+1) % interval == 0:
                learning_curve[index] /= interval
                acc = round(learning_correct / total, 4)
                train_accuracy_curve.append(acc) # add acc
                learning_correct = 0
                total = 0
                print("Train:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                        epoch+1, epochs_n, i+1, len(train_loader), learning_curve[index], acc))
                index += 1
                learning_curve.append(0) # add loss
        
        # display.clear_output(wait=True) # 清除之前的输出，使之即时动态显示

        # validation
        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        model.eval() # switch to evaluation mode
        with torch.no_grad():
            for j, data in enumerate(val_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().sum().cpu().numpy()

                loss_val += loss.item()

            acc = correct_val / total_val
            if acc > max_val_accuracy: # 更新最大正确率
                max_val_accuracy = acc
                max_val_accuracy_epoch = epoch
            #valid_losscurve.append(round(loss_val/len(val_loader), 4)) # 记录损失函数曲线
            val_accuracy_curve[epoch] = round(acc, 4) # 记录准确率曲线
            print("\nValid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}\n".format(
                epoch+1, epochs_n, j+1, len(val_loader), loss_val/len(val_loader), correct_val / total_val))

    print('Training finish, takes {:.2f} seconds'.format(time.time() - start))
    print('The max validation accuracy is: {:.2%}, it reached at {} epoch.\n'.\
          format(max_val_accuracy, max_val_accuracy_epoch))
    
    return learning_curve, grads_diff, train_accuracy_curve, val_accuracy_curve


# Train your model
# feel free to modify
if __name__ == '__main__':
    epo = 20
    loss_save_path = './loss_landscape'
    grad_save_path = './grad_landscape'
    acc_save_path = './acc_landscape'
    
    #set_random_seeds(seed_value=2020, device=device)
    for lr in [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]: # 不能取1e-2，此时训练失败，损失爆炸
        model1 = VGG19()
        model2 = VGG19_BN()

        optimizer1 = torch.optim.Adam(model1.parameters(), lr = lr)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr = lr)
        criterion = nn.CrossEntropyLoss()
        
        result = train(model1, optimizer1, criterion, train_loader, val_loader, epochs_n=epo)
        result_bn = train(model2, optimizer2, criterion, train_loader, val_loader, epochs_n=epo)
        loss, grads_diff, train_acc, val_acc = result
        loss_bn, grads_diff_bn, train_acc_bn, val_acc_bn = result_bn
        
        # save loss and grads difference
        if not os.path.exists(loss_save_path):
            os.mkdir(loss_save_path)
        with open(os.path.join(loss_save_path, 'loss.txt'), 'a') as f:
            f.write(str(loss) + '\n')
        with open(os.path.join(loss_save_path, 'bn_loss.txt'), 'a') as f:
            f.write(str(loss_bn) + '\n')
        
        if not os.path.exists(grad_save_path):
            os.mkdir(grad_save_path)
        with open(os.path.join(grad_save_path, 'grads.txt'), 'a') as f:
            f.write(str(grads_diff) + '\n')
        with open(os.path.join(grad_save_path, 'bn_grads.txt'), 'a') as f:
            f.write(str(grads_diff_bn) + '\n')
        
        # save train, val accuracy
        if not os.path.exists(acc_save_path):
            os.mkdir(acc_save_path)
        with open(os.path.join(acc_save_path, 'acc.txt'), 'a') as f:
            f.write('train_acc' + str(train_acc) + '\n')
            f.write('val_acc' + str(val_acc) + '\n')
            f.write('train_acc_bn' + str(train_acc_bn) + '\n')
            f.write('val_acc_bn' + str(val_acc_bn) + '\n')












