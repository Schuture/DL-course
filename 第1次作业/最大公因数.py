def gcd(a, b):
    if a == 0:
        return b
    if b == 0:
        return a
    n = min(abs(a), abs(b))
    for i in range(n+1, 0, -1):
        if a % i == 0 and b % i == 0:
            return i

print('GCD(3, 5) =', gcd(3, 5))
print('GCD(6, 3) =', gcd(6, 3))
print('GCD(-2, 6) =', gcd(-2, 6))
print('GCD(0, 3) =', gcd(0, 3))

