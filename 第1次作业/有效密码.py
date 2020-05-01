import string

def valid(password):
    upper = False
    lower = False
    digit = False
    sign = False
    for s in password:
        if s in string.ascii_uppercase:
            upper = True
        if s in string.ascii_lowercase:
            lower = True
        if s in string.digits:
            digit = True
        if s in '$#@':
            sign = True
    return upper and lower and digit and sign

def findValidPasswords(passwords):
    ret = []
    passwords = passwords.split(',')
    for password in passwords:
        if 6 <= len(password) <= 12 and valid(password): # 短路，防止遇到太长密码的多余计算
            ret.append(password)
    return ','.join(ret)

passwords = 'ABd1234@1,aF1#,2w3E*,2We3345,UvwXY123@#'
print('All passwords: ', passwords)
print('Valid passwords: ', findValidPasswords(passwords))

