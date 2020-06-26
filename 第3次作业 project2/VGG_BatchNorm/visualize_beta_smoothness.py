import matplotlib.pyplot as plt

def VGG_Beta_Smooth(curve1, curve2):
    plt.figure(figsize=(15,12))
    plt.style.use('ggplot')
    plt.plot(curve1)
    plt.plot(curve2)
    
    plt.xlabel('step')
    plt.ylabel('max difference in gradient')
    plt.title('effective beta smooth')
    plt.legend(['Standard VGG', 'Standard VGG + BatchNorm'])
    
    plt.show()
    
if __name__ == '__main__':
    with open('beta.txt') as f:
        beta_smooth = f.readline()
        beta_smooth_bn = f.readline()
        
    beta_smooth = [float(s) for s in beta_smooth[1:-2].split(',')]
    beta_smooth_bn = [float(s) for s in beta_smooth_bn[1:-2].split(',')]
    
    VGG_Beta_Smooth(beta_smooth[3:], beta_smooth_bn[3:])