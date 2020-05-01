def add(A, B):
    n, m = len(A), len(A[0])
    ret = [[0 for j in range(m)] for i in range(n)]
    for i in range(n):
        for j in range(m):
            ret[i][j] = A[i][j] + B[i][j]
    return ret


def subtract(A, B):
    n, m = len(A), len(A[0])
    ret = [[0 for j in range(m)] for i in range(n)]
    for i in range(n):
        for j in range(m):
            ret[i][j] = A[i][j] - B[i][j]
    return ret


def scalar_multiply(A, num):
    n, m = len(A), len(A[0])
    ret = [[0 for j in range(m)] for i in range(n)]
    for i in range(n):
        for j in range(m):
            ret[i][j] = num * A[i][j]
    return ret


def multiply(A, B):
    n, m, l = len(A), len(A[0]), len(B[0])
    ret = [[0 for j in range(l)] for i in range(n)] # n*m x m*l => n*l
    for i in range(n):
        for j in range(l):
            for k in range(m):
                tmp = A[i][k] * B[k][j]
                ret[i][j] += tmp
    return ret


def identity(n):
    ret = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        ret[i][i] = 1
    return ret


def transpose(A):
    n = len(A) # A必须是方阵
    for i in range(n):
        for j in range(i+1, n):
            A[i][j], A[j][i] = A[j][i], A[i][j]
    return A


def step0(m):
    n = len(m)
    l = []
    for i in range(0,n):
        l.append([])
        for j in range(0,n):
            if i == j:
                l[i].append(1)
            else:
                l[i].append(0)
    return l


# 以下几个函数都是高斯消元法求逆矩阵
def step1(m):
    n = len(m)
    """交换操作记录数组 swap"""
    swap = []
    l = []
    for i in range(0,n):
        swap.append(i)
        l.append([])
        for j in range(0,n):
            l[i].append(0)

    """对每一列进行操作"""
    for i in range(0,n):
        max_row = m[i][i]
        row = i
        for j in range(i,n):
            if m[j][i] >= max_row:
                max_row = m[j][i]
                row = j
        swap[i] = row

        """交换"""
        if row != i:
            for j in range(0,n):
                m[i][j],m[row][j] = m[row][j],m[i][j]

        """消元"""
        for j in range(i+1,n):
            if m[j][i] != 0:
                l[j][i] = m[j][i] / m[i][i]
                for k in range(0,n):
                    m[j][k] = m[j][k] - (l[j][i] * m[i][k])

    return (swap,m,l)


def step2(m):
    n = len(m)
    long = len(m)-1
    l = []
    for i in range(0,n):
        l.append([])
        for j in range(0,n):
            l[i].append(0)

    for i in range(0,n-1):
        for j in range(0,long-i):
            if m[long-i-j-1][long-i] != 0 and m[long-i][long-i] != 0:
                l[long-i-j-1][long-i] = m[long-i-j-1][long-i] / m[long-i][long-i]
                for k in range(0,n):
                    m[long-i-j-1][k] = m[long-i-j-1][k] - l[long-i-j-1][long-i] * m[long-i][k]

    return (m,l)


def step3(m):
    n = len(m)
    l = []
    for i in range(0,n):
        l.append(m[i][i])
    return l


def inverse(matrix): # 利用step1/2/3来求逆矩阵
    n = len(matrix)
    new = step0(matrix)
    (swap,matrix1,l1) = step1(matrix)
    (matrix2,l2) = step2(matrix1)
    l3 = step3(matrix2)
    for i in range(0,n):
        if swap[i] != i:
            new[i],new[swap[i]] = new[swap[i]],new[i]
        for j in range(i+1,n):
            for k in range(0,n):
                if l1[j][i] != 0:
                    new[j][k] = new[j][k] - l1[j][i] * new[i][k]   
    for i in range(0,n-1):
        for j in range(0,n-i-1):
            if l2[n-1-i-j-1][n-1-i] != 0:
                for k in range(0,n):
                    new[n-1-i-j-1][k] = new[n-1-i-j-1][k] - l2[n-1-i-j-1][n-i-1] * new[n-1-i][k]
    for i in range(0,n):
        for j in range(0,n):
            new[i][j] = new[i][j] / l3[i]
            
    return new


A = [[12, 10], [3, 9]]
B = [[3, 4], [7, 4]]
C = [[11,12,13,14], [21,22,23,24], [31,32,33,34], [41,42,43,44]]
D = [[3, 0, 2], [2, 0, -2], [0, 1, 1]]
print('A:', A)
print('B:', B)
print('C:', C)
print('D:', D)

print('\nA + B:')
print(add(A, B))
print('A - B:')
print(subtract(A, B))
print('B x 3:')
print(scalar_multiply(B, 3))
print('A x B:')
print(multiply(A, B))
print('对角矩阵：')
print(identity(3))
print('C的转置：')
print(transpose(C))
print('D的逆矩阵：')
print(inverse(D))

