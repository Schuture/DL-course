import matplotlib.pyplot as plt


def plot_acc_landscape(train_acc, val_acc, train_acc_bn, val_acc_bn):
    ratio = len(train_acc) / len(val_acc)
    x1 = [i/ratio for i in range(len(train_acc))]
    x2 = [i+1 for i in range(len(val_acc))]
    
    plt.figure(figsize=(16, 12))
    plt.style.use('ggplot')
    
    plt.subplot(211)
    plt.plot(x1, train_acc)
    plt.plot(x2, val_acc)
    plt.title('The training and validation accuracy curve without BN')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['training accuracy', 'validation accuracy'], loc='lower center')
    
    plt.subplot(212)
    plt.plot(x1, train_acc_bn)
    plt.plot(x2, val_acc_bn)
    plt.title('The training and validation accuracy curve with BN')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['training accuracy', 'validation accuracy'], loc='lower center')
    
    plt.show()
    

if __name__ == '__main__':
    with open('acc.txt') as f:
        train_acc = [float(acc) for acc in f.readline()[11:-2].split(',')]
        val_acc = [float(acc) for acc in f.readline()[9:-2].split(',')]
        train_acc_bn = [float(acc) for acc in f.readline()[14:-2].split(',')]
        val_acc_bn = [float(acc) for acc in f.readline()[12:-2].split(',')]

    plot_acc_landscape(train_acc, val_acc, train_acc_bn, val_acc_bn)















