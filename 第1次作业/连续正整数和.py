def positiveSums(N):
    ret = []
    i = 1
    while N - i >= 0:
        N -= i
        if N % (i+1) == 0:
            start = N // (i+1)
            ret.append([start + j for j in range(i+1)])
        i += 1
    return ret

print('100可拆成以下数组和：', positiveSums(100))
print('1000可拆成以下数组和：', positiveSums(1000))


