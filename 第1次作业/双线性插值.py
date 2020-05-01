import numpy as np

def BilinearInterpolation(A, coordinate):
    '''
    Bilinear interpolation of an array
    Inputs:
        A: a numpy ndarray
        coordinate: a tuple consists of 2 numbers
    Outputs:
        ret: the pixel value at the target coordinate
    '''
    x, y = coordinate
    x -= 1
    y -= 1
    if x - int(x) < 1e-6 and y - int(y) < 1e-6: # 本就在格点
        return A[int(x), int(y)]
    elif x - int(x) < 1e-6: # 在水平边上
        left = int(np.floor(y))
        right = int(np.ceil(y))
        return A[int(x), left] + (A[int(x), right] - A[int(x), left]) * (y - left)
    elif y - int(y) < 1e-6: # 在垂直边上
        up = int(np.floor(x))
        down = int(np.ceil(x))
        return A[up, int(y)] + (A[down, int(y)] - A[up, int(y)]) * (x - up)
    else: # 在格点之间
        left = int(np.floor(y))
        right = int(np.ceil(y))
        up = int(np.floor(x))
        down = int(np.ceil(x))
        left_pixel = A[up, left] + (A[down, left] - A[up, left]) * (x - up)
        right_pixel = A[up, right] + (A[down, right] - A[up, right]) * (x - up)
        center_pixel = left_pixel + (right_pixel - left_pixel) * (y - left)
        return center_pixel
    
A = np.array([[110, 120, 130], [210, 220, 230], [310, 320, 330]])
coordinate = (1, 1)
print('坐标{}处插值像素为：{}'.format(coordinate, BilinearInterpolation(A, coordinate)))
coordinate = (2.5, 2.5)
print('坐标{}处插值像素为：{}'.format(coordinate, BilinearInterpolation(A, coordinate)))
coordinate = (1.7, 2.4)
print('坐标{}处插值像素为：{}'.format(coordinate, BilinearInterpolation(A, coordinate)))


