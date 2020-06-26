import re
import matplotlib.pyplot as plt

epochs = []
losses = []
det_losses = []
rec_losses = []
with open('nohup.out') as f:
    while True:
        line = f.readline()
        if not line:
            break
        if re.search('    epoch.*:.*\w', line):
            epoch = int(line.split()[2])
            print('Found log of epoch {}'.format(epoch))
            epochs.append(epoch)
        elif re.search('    loss.*:.*\w', line):
            print('Found loss of epoch {}'.format(epoch))
            losses.append(float(line.split()[2]))
        elif re.search('    det_loss.*:.*\w', line):
            print('Found detection loss of epoch {}'.format(epoch))
            det_losses.append(float(line.split()[2]))
        elif re.search('    rec_loss.*:.*\w', line):
            print('Found detection loss of epoch {}'.format(epoch))
            rec_losses.append(float(line.split()[2]))

print(epochs, '\n')
print([round(loss, 4) for loss in losses], '\n')
print([round(det_loss, 4) for det_loss in det_losses], '\n')
print([round(rec_loss, 4) for rec_loss in rec_losses], '\n')

plt.figure(figsize = (16, 12))
plt.style.use('ggplot')
plt.plot(epochs, losses)
plt.plot(epochs, det_losses)
plt.plot(epochs, rec_losses)
plt.title('FOTS loss curves', fontsize=24)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('loss', fontsize=20)
plt.legend(['Loss', 'detection loss', 'recognization loss'], fontsize=20)
plt.tick_params(labelsize=16)
plt.savefig('loss_curve.jpg')
plt.show()
















