import matplotlib.pyplot as plt

FILE_PATH = 'D:/学习/课程/大数据/深度学习和神经网络/作业/第4次作业 project3/project3/codes/'
with open(FILE_PATH+'2020_06_22_14_11_00_88.09%'+'.txt') as f:
    line = f.readline()
    while True:
        if not line:
            break
        elif line.startswith('Training loss'):
            training_loss = f.readline()
            training_loss = [float(loss) for loss in training_loss[1:-2].split(',')]
        elif line.startswith('Training acc'):
            training_acc = f.readline()
            training_acc = [float(acc) for acc in training_acc[1:-2].split(',')]
        elif line.startswith('Validation loss'):
            validation_loss = f.readline()
            validation_loss = [float(loss) for loss in validation_loss[1:-2].split(',')]
        elif line.startswith('Validation acc'):
            validation_acc = f.readline()
            validation_acc = [float(acc) for acc in validation_acc[1:-2].split(',')]
        line = f.readline()

# 一个epoch可能会保存多个训练acc/loss，不管保存了几个，我们都只取一个来可视化
#training_loss = training_loss[::9]
#training_acc = training_acc[::9]

n = len(training_loss)

plt.figure(figsize=(16, 16))

plt.subplot(211)
m1 = plt.plot(list(range(1, n+1)), training_loss)
m2 = plt.plot(list(range(1, n+1)), validation_loss)
plt.title('Loss vs time', fontsize=24)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('loss', fontsize=20)
plt.tick_params(labelsize=13)
plt.legend(["Training loss", "Validation loss"], loc='upper right', fontsize=20)

plt.subplot(212)
m3 = plt.plot(list(range(1, n+1)), training_acc)
m4 = plt.plot(list(range(1, n+1)), validation_acc)
plt.title('Accuracy vs time', fontsize=24)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('acc', fontsize=20)
plt.tick_params(labelsize=13)
plt.legend(["Training accuracy", "Validation accuracy"], loc='lower right', fontsize=20)

plt.show()



