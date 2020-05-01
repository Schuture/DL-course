import numpy as np
np.random.seed(2020)

# 1 笛卡尔坐标转极坐标
cartesian = np.random.randn(10, 2) # 笛卡尔坐标下的10个点
print('cartesian:\n', cartesian)
# 模长，角度（弧度制）
r = np.linalg.norm(cartesian, axis = 1)
theta = np.arctan(cartesian[:,1] / cartesian[:,0])
theta += np.pi * (cartesian[:,0] < 0) # 第二三象限要加上pi
theta += 2 * np.pi * ((cartesian[:, 0] > 0) & (cartesian[:, 1] < 0)) # 第四象限加2pi
polar = np.vstack((r, theta)).T
print('\npolar:\n', polar)

# 2 2D array subclass such that Z[i,j] == Z[j,i]
X = np.random.rand(5, 5)
X = np.triu(X)
X += X.T - np.diag(X.diagonal())
print('\nSymmetric matrix X: \n', np.round(X, 3))
print('\n')

# 3 计算点到线的距离
P = np.array([[0, 0], [1, 0], [0, 1]])
P0 = np.array([[0, 2], [0, 2], [0, 2]])
P1 = np.array([[2, 0], [1, 0], [-1, 0]])
for i in range(len(P0)):
    # 找到直线 y = kx+b / kx-y+b = 0
    p0, p1 = P0[i], P1[i]
    k = (p1[1] - p0[1]) / (p1[0] - p0[0])
    b = p0[1] - k*p0[0]
    for j in range(len(P)):
        p = P[j]
        distance = round(abs(k*p[0]-p[1]+b) / np.sqrt(k**2+1), 3)
        print('由{}、{}两点组成的直线到点{}距离为{}'.format(p0, p1, p, distance))