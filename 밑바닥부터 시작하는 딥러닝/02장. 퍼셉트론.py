import numpy as np

# 2.3 퍼셉트론 구현하기
# 2.3.1 간단한 구현부터
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

print(AND(0, 0)) # 0출력
print(AND(1, 0)) # 0출력
print(AND(0, 1)) # 0출력
print(AND(1, 1)) # 1출력

# 2.3.2 가중치와 편향 도입
x = np.array([0,1])
w = np.array([0.5, 0.5])
b = -0.7
print(w*x)
print(np.sum(w*x) + b)

# 2.3.3 가중치와 편향 구현하기
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    print('tmp =', tmp)
    if tmp <= 0:
        return 0
    else:
        return 1

print(AND(0, 0)) # 0출력
print(AND(1, 0)) # 0출력
print(AND(0, 1)) # 0출력
print(AND(1, 1)) # 1출력

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    print('tmp =', tmp)
    if tmp <= 0:
        return 0
    else:
        return 1

print(NAND(0, 0)) # 1출력
print(NAND(1, 0)) # 1출력
print(NAND(0, 1)) # 1출력
print(NAND(1, 1)) # 0출력

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1.0, 1.0])
    b = -0.5
    tmp = np.sum(w * x) + b
    print('tmp =', tmp)
    if tmp <= 0:
        return 0
    else:
        return 1

print(OR(0, 0)) # 1출력
print(OR(1, 0)) # 1출력
print(OR(0, 1)) # 1출력
print(OR(1, 1)) # 1출력

# 2.4 퍼셉트론의 한계
# AND, NAND, OR는 선형구조이기 때문에 단층으로 구현이 가능하지만 XOR는 비선형 구조이므로 다층 퍼셉트론으로 구현해야한다.

# 2.5.2 XRO 게이트 구현하기
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(XOR(0, 0)) # 0 출력
print(XOR(1, 0)) # 1 출력
print(XOR(0, 1)) # 1 출력
print(XOR(1, 1)) # 0 출력
