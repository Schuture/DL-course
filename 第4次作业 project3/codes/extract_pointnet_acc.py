import matplotlib.pyplot as plt

train_acc = []
valid_acc = []
train_acc_this_epoch = []
with open('nohup.out') as f:
    while True:
        line = f.readline()
        if not line:
            break
        if line.startswith('Valid:'):
            acc_str = line[-10:].split(':')[1][:-2]
            valid_acc.append(float(acc_str))
            train_acc.append(train_acc_this_epoch[-1])
            train_acc_this_epoch = []
        elif line.startswith('Training:'):
            acc_str = line[-10:].split(':')[1][:-2]
            train_acc_this_epoch.append(float(acc_str))

epochs = list(range(1, len(valid_acc)+1))
max_train_acc = max(train_acc)
max_val_acc = max(valid_acc)
print('Trained {} epochs.'.format(len(valid_acc)))
print('The max train accuracy is: {}, reached at {} epoch.'.format(max_train_acc, train_acc.index(max_train_acc)))
print('The max valid accuracy is: {}, reached at {} epoch.'.format(max_val_acc, valid_acc.index(max_val_acc)))

plt.figure(figsize = (16, 12))
plt.style.use('ggplot')
plt.plot(epochs, train_acc)
plt.plot(epochs, valid_acc)
plt.title('PointNet accuracy curves', fontsize=24) # 或者 PointNet++
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('accuracy (%)', fontsize=20)
plt.legend(['Training accuracy', 'Testing accuracy'], fontsize=20, loc='lower right')
plt.tick_params(labelsize=16)
plt.savefig('pointnet2_acc_curve.jpg')
plt.show()
















