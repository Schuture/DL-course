def cartesianProduct(A):
    '''
    Get the Cartesian product of a series of arrays
    Input:
        A: a list contains several lists, sub-lists contains integers
    Outpus:
        C_product: a list contains the cartesian products
    '''
    if len(A) == 1:
        return [[a] for a in A[0]]
    nums = A.pop(0) # 第一个数组
    post_product = cartesianProduct(A)
    C_product = []
    for num in nums: # 以num开头
        product = []
        for p in post_product:
            product.append([num] + p)
        C_product.extend(product)
    return C_product

A = [[1, 2, 3], [4, 5], [6, 7]]
print('{} 的笛卡尔积为：'.format(A))
print(cartesianProduct(A))


