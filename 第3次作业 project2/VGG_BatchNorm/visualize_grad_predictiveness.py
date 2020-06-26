import matplotlib.pyplot as plt
import numpy as np


def VGG_Grad_Pred(x, min_curve, max_curve, bn_min_curve, bn_max_curve):
    plt.figure(figsize=(15, 12))
    plt.style.use('fivethirtyeight')
    
    plt.plot(x, min_curve, color="#59aa6c")
    plt.plot(x, max_curve, color="#59aa6c")
    plt.plot(x, bn_min_curve, color="#c44e52")
    plt.plot(x, bn_max_curve, color="#c44e52")
    
    p1 = plt.fill_between(x, min_curve, max_curve, facecolor="#9fc8ac")
    p2 = plt.fill_between(x, bn_min_curve, bn_max_curve, facecolor="#d69ba1")
                     
    plt.title('Gradient Predictiveness')
    plt.ylabel('gradient difference')
    plt.xlabel('step')
    
    plt.legend([p1, p2], ['Standard VGG', 'Standard VGG + BatchNorm'])
    
    plt.show()
    

if __name__ == '__main__':
    grads_diff = []
    with open('grads.txt') as f:
        while True:
            grad_diff = f.readline()
            if not grad_diff:
                break
            grad_diff = [float(loss) for loss in grad_diff[1:-2].split(',')]
            grads_diff.append(grad_diff)
    
    bn_grads_diff = []
    with open('bn_grads.txt') as f:
        while True:
            bn_grad_diff = f.readline()
            if not bn_grad_diff:
                break
            bn_grad_diff = [float(loss) for loss in bn_grad_diff[1:-2].split(',')]
            bn_grads_diff.append(bn_grad_diff)

    grads_diff = np.array(grads_diff)[:,3:]
    bn_grads_diff = np.array(bn_grads_diff)[:,3:]
    
    min_curve = grads_diff.min(0)
    max_curve = grads_diff.max(0)
    bn_min_curve = bn_grads_diff.min(0)
    bn_max_curve = bn_grads_diff.max(0)
    x = [i for i in range(len(max_curve))]
    VGG_Grad_Pred(x, min_curve, max_curve, bn_min_curve, bn_max_curve)















