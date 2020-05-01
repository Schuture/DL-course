import numpy as np
np.random.seed(2020)

def subPart(A, position, shape, fill = 0):
    n, m = A.shape
    x, y = position[0] - 1, position[1] - 1
    height, width = shape
    left = y - (width - 1) // 2
    right = left + width
    up = x - (height - 1) // 2
    down = up + height
    
    if 0 <= left and right <= n and 0 <= up and down <= m: # 完全包含
        return A[up:down, left:right]
    
    pad_width = max([0, -left, -up, right-m, down-n]) # 计算填充宽度
    A = np.pad(A, pad_width = pad_width, mode = 'constant', constant_values = 0)
    return A[up+pad_width:down+pad_width, left+pad_width:right+pad_width]

A = np.random.randint(0, 10, (5, 5))
shape = (4, 3)
position = (1, 1)
print('原矩阵：')
print(A)
print('\n形状{}，中心{}的子矩阵如下：'.format(shape, position))
print(subPart(A, position, shape))

