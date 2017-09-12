# 문제 27. 아래의 식을 python으로 구현하시오
import numpy as np
x = np.array([0, 1])
w = np.array([0.5, 0.5])
print(np.sum(x*w))

# 문제 28. 위의 식에서 편향을 더하여 완성한식을 구현하시오
x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7 # 편향
print(np.sum(x*w) + b)

# 문제 29. AND게이트를 구현하시오!
import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0.7
    tmp = np.sum(x*w)
    if tmp <= b:
        return 0
    else:
        return 1

print(AND(0,0)) # 0 출력
print(AND(1,0)) # 0 출력
print(AND(0,1)) # 0 출력
print(AND(1,1)) # 1 출력

# 문제 30(점심시간 문제). 위의 함수에 편향을 포함하여 AND게이트 함수를 구현하시오
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print(AND(0,0)) # 0 출력
print(AND(1,0)) # 0 출력
print(AND(0,1)) # 0 출력
print(AND(1,1)) # 1 출력


#########################
def ex(x0, x1, x2):
    x = np.array([x0, x1, x2])
    w = np.array([0.4, 0.3, 0.1])
    tmp = np.sum(x*w)
    print(tmp)
    if tmp < 0:
        return 0
    else:
        return 1

print(ex(-1,0,0))
print(ex(-1,0,1))
print(ex(-1,1,0))
print(ex(-1,1,1))


# 문제 31. NAND게이트를 함수로 구현하시오!
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print(NAND(0,0)) # 1을 출력
print(NAND(1,0)) # 1을 출력
print(NAND(0,1)) # 1을 출력
print(NAND(1,1)) # 0을 출력

# 문제 32. OR게이트를 함수로 구현하시오!
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print(OR(0,0)) # 0을 출력
print(OR(1,0)) # 1을 출력
print(OR(0,1)) # 1을 출력
print(OR(1,1)) # 1을 출력

# 문제 33. XOR게이트를 함수로 구현하시오
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(XOR(0,0)) # 0을 출력
print(XOR(1,0)) # 1을 출력
print(XOR(0,1)) # 1을 출력
print(XOR(1,1)) # 0을 출력

-1*0.25 + 0*0.4 + 1*0.15

