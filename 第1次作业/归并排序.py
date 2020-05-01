import time
import random

n = 1000
m = 1000

def merge(A,p,q,r): #包含首尾A[p],A[r]
    n1 = q-p+1
    n2 = r-q
    L = [0] * (n1+1)
    R = [0] * (n2+1)
    for i in range(n1): #第一段A[p...q]
        L[i] = A[p+i]
    for j in range(n2): #第二段A[q+1...r]
        R[j] = A[q+1+j]
    L[-1] = float('inf')   
    R[-1] = float('inf')
    i = 0
    j = 0
    for k in range(p,r+1):
        if L[i]<=R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1
            
def MergeSort(A,p,r):
    if p<r:
        q = int((p+r)//2)
        MergeSort(A,p,q)
        MergeSort(A,q+1,r)
        merge(A,p,q,r)

A = list(range(20))
random.shuffle(A)
MergeSort(A,0,19)
print(A)
start = time.time()
for i in range(n):
    A = list(range(m))
    random.shuffle(A)
    MergeSort(A,0,m-1)
end = time.time()
print('MergeSort consumes {} seconds to sort {} lists with length {}'.format(end-start,n,m),'\n',
      'Average {} seconds'.format((end-start)/n))
