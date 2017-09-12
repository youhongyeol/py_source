import numpy as np
import matplotlib.pylab as plt

# 3.2.2 계단 함수 구현하기
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

print(step_function(3))
print(step_function(np.array([1, 2])))
# 위의 step_function으로는 numpy 배열은 처리할 수 없다.

def step_function(x):
    y = x > 0
    return y.astype(np.int)

# astype(np.int)에 대해 알아보기
x = np.array([-1, 1, 2])
y = x > 0
y # array([False,  True,  True], dtype=bool)
y = y.astype(np.int)
y # array([0, 1, 1])

def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5, 5, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축의 범위 지정
plt.show()

# 3.2.4 시그모이드 함수 구현하기
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1, 1, 2])
print(sigmoid(x))

# numpy의 브로드캐스트!!
t = np.array([1, 2, 3])
print(1 + t)

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# 3.2.7 ReLU 함수
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu(x)
plt.plot(x, y)
plt.xlim(-6, 6)
plt.ylim(-1, 5)
plt.show()

# 3.3.1 다차원 배열
A = np.array([1,2,3,4])
print(A)
np.ndim(A) # 배열의 차원수: 1차원
A.shape # 배열의 형상: (4, )
A.shape[0] # 4

B = np.array([[1,2], [3,4], [5,6]])
print(B)
np.ndim(B) # 2차원
B.shape # (3, 2) 3행 2열을 나타낸다.

# 3.3.2 행렬의 내적(행렬 곱)
A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])
np.dot(A,B)
# array([[19, 22],
#        [43, 50]])
np.dot(B,A)
# array([[23, 34],
#        [31, 46]])
# 순서에 따라 다른 값이 출력된다.

A = np.array([[1,2,3], [4,5,6]])
B = np.array([[1,2], [3,4], [5,6]])
np.dot(A,B)
# array([[22, 28],
#        [49, 64]])
np.dot(B,A)
# array([[ 9, 12, 15],
#        [19, 26, 33],
#        [29, 40, 51]])

A = np.array([[1,2], [3,4], [5,6]])
B = np.array([7,8])
np.dot(A,B) # (3, 2) * (2, )
# array([23, 53, 83])

# 3.3.3 신경망의 내적
X = np.array([1,2])
W = np.array([[1,3,5], [2,4,6]])
Y = np.dot(X,W)
print(Y)

X = np.array([1,0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
b1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X,W1) + b1
Z1 = sigmoid(A1) # 활성화 함수 통과
print(A1)
print(Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
b2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + b2
Z2 = sigmoid(A2)
print(A2)
print(Z2)

def identity_function(x):
    return x

W3 = np.array([[0.1,0.3], [0.2,0.4]])
b3 = np.array([0.1,0.2])
A3 = np.dot(Z2, W3) + b3
Z3 = identity_function(A3)
print(A3)
print(Z3)

# 3.4.3 구현정리
def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3], [0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1, 0.5])
y = forward(network, x)
print(y)

# 3.5.1 항등 함수와 소프트맥스 함수 구현하기
a = np.array([0.3,2.9,4.0])
exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# 3.5.2 소프트맥스 함수 구현 시 주의점
np.exp(1000)
a = np.array([1010, 1000, 990])
np.exp(a) / np.sum(np.exp(a)) # 제대로 계산되지 않는다.

c = np.max(a)
a - c
np.exp(a-c) / np.sum(np.exp(a-c))

# 오버플로를 해결한 softmax 함수
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(np.exp(a-c))
    y = exp_a / sum_exp_a
    return y
print(softmax(a))

# 3.5.3 소프트맥스 함수의 특징
a = np.array([0.3,2.9,4.0])
y = softmax(a)
print(y)
np.sum(y) # 결과의 합은 1이 된다.



