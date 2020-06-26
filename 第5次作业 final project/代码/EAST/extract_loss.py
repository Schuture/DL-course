import re
import matplotlib.pyplot as plt

epoch = 1
epochs = []
det_losses = []
with open('nohup.out') as f:
    while True:
        line = f.readline()
        if not line:
            break
        elif re.search('epoch_loss is \w', line):
            print('Found detection loss of epoch {}'.format(epoch))
            epochs.append(epoch)
            det_losses.append(float(line.split()[2][:-2]))
            epoch += 1

print(epochs, '\n')
print([round(det_loss, 4) for det_loss in det_losses], '\n')

plt.figure(figsize = (16, 12))
plt.style.use('ggplot')
plt.plot(epochs, det_losses)
plt.title('EAST detection loss curve', fontsize=24)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('loss', fontsize=20)
plt.legend(['detection loss'], fontsize=20)
plt.tick_params(labelsize=16)
plt.savefig('loss_curve.jpg')
plt.show()
















