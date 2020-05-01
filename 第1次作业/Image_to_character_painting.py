from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为ndarray
img = Image.open('panda.jpg').convert('RGB').resize((70, 70))
img = np.array(img)
n, m = img.shape[:2]

# 显示原图以及转换后的灰度图
plt.figure(figsize = (18, 10))
plt.subplot(121)
plt.imshow(img)

img = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
plt.subplot(122)
plt.imshow(img, cmap=plt.cm.gray)

# 转换为character painting
char = 'ILoveComputerVision'
painting = list()
k, l = 0, len(char)
for i in range(n):
    row = ''
    for j in range(m):
        if img[i][j] < 200:
            row += char[k%l]
            k += 1
        else:
            row += ' '
    painting.append(row)

with open('character_painting.txt', 'w') as f:
    for i in range(len(painting)):
        f.write(painting[i] + '\n')

print(painting)








