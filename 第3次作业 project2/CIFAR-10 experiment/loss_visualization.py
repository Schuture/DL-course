import matplotlib.pyplot as plt

FILE_PATH = 'D:/学习/课程/大数据/深度学习和神经网络/作业/第3次作业 project2/CIFAR-10 experiment/'
with open(FILE_PATH+'2020-05-07_13_51_26_94.63%'+'.txt') as f:
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
        
training_loss = training_loss[::9]
training_acc = training_acc[::9]

n = len(training_loss)

plt.figure(figsize=(12, 12))

plt.subplot(211)
m1 = plt.plot(list(range(1, n+1)), training_loss)
m2 = plt.plot(list(range(1, n+1)), validation_loss)
plt.title('Loss vs time')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(["Training loss", "Validation loss"], loc='upper right')

plt.subplot(212)
m3 = plt.plot(list(range(1, n+1)), training_acc)
m4 = plt.plot(list(range(1, n+1)), validation_acc)
plt.title('Accuracy vs time')
plt.xlabel('Epoch')
plt.ylabel('acc')
plt.legend(["Training accuracy", "Validation accuracy"], loc='lower right')

plt.show()



