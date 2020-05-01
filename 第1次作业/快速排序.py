import random
import time

n = 1000
m = 1000

#快速排序,直接调用，原地排序
class QuickSort(object):
    def __init__(self,A):
        length = len(A)
        self.quicksort(A,0,length-1)
        
    def partition1(self,A,p,r): # A[p,r]包含首尾，双指针同向
        x = A[r] # 最右边那个数
        i = p # 将j指针第一次遇到的小于x的数字搬到这里
        for j in range(p,r): # 希望A[j]比x大，所以遇到比x小的就往最左边没搬过的位置搬
            if A[j]<x:
                A[i],A[j] = A[j],A[i]
                i += 1 # i记录下一次要搬的位置
        A[i],A[r] = A[r],A[i] # 此时i指着第一个大于等于x的数的位置
        return i
    
    def partition2(self, A, p, r): # i，j对向
        x = A[p]
        i, j = p+1, r
        while True:
            while A[i] < x:
                i += 1
                if i > r:
                    break
            while A[j] > x:
                j -= 1
                if j < p+1:
                    break
            if i < j:
                A[i], A[j] = A[j], A[i]
            else:
                A[j], A[p] = A[p], A[j]
                return j
    
    def quicksort(self,A,p,r):
        if p<r:
            q = self.partition2(A,p,r)
            self.quicksort(A,p,q-1)
            self.quicksort(A,q+1,r)
        return A

A = list(range(20))
random.shuffle(A)
QuickSort(A)
print(A)
start = time.time()
for i in range(n):
    A = list(range(m))
    random.shuffle(A)
    QuickSort(A)
end = time.time()
print('QuickSort consumes {} seconds to sort {} lists with length {}'.format(end-start,n,m),'\n',
      'average {} seconds'.format((end-start)/n))