import random
import time

n = 10
m = 1000

# 冒泡排序
def BubbleSort(A):
    for i in range(len(A)):
        for j in range(0, len(A)-i-1): # 排好一个数以后，最后一个数字必定最大
            if A[j] > A[j+1]:
                A[j],A[j+1] = A[j+1],A[j]
             
A = list(range(20))
random.shuffle(A)
BubbleSort(A)
print(A)

start = time.time()
for i in range(n):
    A = list(range(m))
    random.shuffle(A)
    BubbleSort(A)
end = time.time()
print('BubbleSort consumes {} seconds to sort {} lists with length {}'.format(end-start,n,m),'\n',
      'average {} seconds'.format((end-start)/n))

