import time,random

n = 100
m = 1000

def ShellSort(A):
    step = int(len(A)/2) #分组步长，例如步长为5就是对第1,6,11,16...个元素排序
    while step > 0:
        # print("---step ---", step)
        #对分组数据进行插入排序
        for index in range(0,len(A)):
            if index + step < len(A):
                current_val = A[index] #先记下来每次大循环走到的第几个元素的值
                if current_val > A[index+step]: #switch
                    A[index], A[index+step] = A[index+step], A[index]

        step = int(step/2) # step每次减半，直到1
    
    else: #把基本排序好的数据再进行一次插入排序就好了
        for index in range(1, len(A)):
            current_val = A[index]  # 先记下来每次大循环走到的第几个元素的值
            position = index
            # 当前元素的左边的紧靠的元素比它大,
            # 要把左边的元素一个一个的往右移一位,
            # 给当前这个值插入到左边挪一个位置出来
            while position > 0 and A[position - 1] > current_val:
                A[position] = A[position - 1]  # 把左边的一个元素往右移一位
                # 只一次左移只能把当前元素一个位置，
                # 还得继续左移只到此元素放到排序好的列表的适当位置为止
                position -= 1  
            
            # 已经找到了左边排序好的列表里不小于current_val的元素的位置,
            # 把current_val放在这里
            A[position] = current_val  
    return A

A = list(range(20))
random.shuffle(A)
print(ShellSort(A))

start = time.time()
for i in range(n):
    A = list(range(m))
    random.shuffle(A)
    ShellSort(A)
end = time.time()
print('BubbleSort consumes {} seconds to sort {} lists with length {}'.format(end-start,n,m),'\n',
      'average {} seconds'.format((end-start)/n))
