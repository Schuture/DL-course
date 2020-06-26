import matplotlib.pyplot as plt
import numpy as np


def plot_loss_landscape(x, min_curve, max_curve, bn_min_curve, bn_max_curve):
    plt.figure(figsize=(15, 12))
    plt.style.use('fivethirtyeight')
    
    plt.plot(x, min_curve, color="#59aa6c")
    plt.plot(x, max_curve, color="#59aa6c")
    plt.plot(x, bn_min_curve, color="#c44e52")
    plt.plot(x, bn_max_curve, color="#c44e52")
    
    p1 = plt.fill_between(x, min_curve, max_curve, facecolor="#9fc8ac")
    p2 = plt.fill_between(x, bn_min_curve, bn_max_curve, facecolor="#d69ba1")
                     
    plt.title('loss landscape')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    
    plt.legend([p1, p2], ['Standard VGG + Dropout', 'Standard VGG + BatchNorm'])
    
    plt.show()
    

if __name__ == '__main__':
    train_curves = []
    with open('dropout_loss.txt') as f:
    #with open('dropout_loss.txt') as f:
        while True:
            training_loss = f.readline()
            if not training_loss:
                break
            training_loss = [float(loss) for loss in training_loss[1:-2].split(',')]
            train_curves.append(training_loss)
            
    bn_train_curves = []
    with open('bn_loss2.txt') as f:
        while True:
            training_loss = f.readline()
            if not training_loss:
                break
            training_loss = [float(loss) for loss in training_loss[1:-2].split(',')]
            bn_train_curves.append(training_loss)
            
    train_curves = np.array(train_curves)
    bn_train_curves = np.array(bn_train_curves)
    
    min_curve = train_curves.min(0)
    max_curve = train_curves.max(0)
    bn_min_curve = bn_train_curves.min(0)
    bn_max_curve = bn_train_curves.max(0)
    x = [i / 10 for i in range(len(max_curve))]
    
    plot_loss_landscape(x, min_curve, max_curve, bn_min_curve, bn_max_curve)















