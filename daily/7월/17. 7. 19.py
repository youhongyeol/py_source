# 문제 35. 파이썬으로 계단함수를 구현하시오
import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(np.int) # true 1, false 0으로 변경한다.

x_data = np.array([-1,0,1])
print(step_function(x_data)) # [0 0 1]

# 문제 36. step_function 함수를 사용해서 계단함수 그래프를 그리시오
import matplotlib.pylab as plt

x = np.arange(-5,5,0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# 문제 37. 아래와 같이 그래프가 출력되게 하시오
def step_function1(x):
    y = x < 0
    return y.astype(np.int) # true 0, false 1으로 변경한다.

x_data = np.array([-1,0,1])
print(step_function1(x_data))

x = np.arange(-5,5,0.1)
y = step_function1(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# 문제 38(점심시간 문제)
x = np.array([-1, 0, 0])

def step_function(x):
    W = np.array([0.3, 0.4, 0.1])
    y = np.sum(x*W) > 0
    return y.astype(np.int)

print(step_function(x))

# 문제 39. sigmoid 함수를 수현하시오
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1, 1, 2])
print(sigmoid(x))

# 문제 40. sigmoid 함수를 그래프로 그리시오
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# 문제 41. 아래와 같이 그래프를 그리시오
def sigmoid1(x):
    return 1 / (1 + np.exp(x))

x = np.arange(-5, 5, 0.1)
y = sigmoid1(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# 문제 42. 계단함수와 sigmoid 함수를 같이 출력하시오
import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(np.int) # true 1, false 0으로 변경한다.

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)
x1 = np.arange(-5, 5, 0.1)
y1 = step_function(x)
plt.plot(x, y)
plt.plot(x1, y1)
plt.ylim(-0.1, 1.1)
plt.show()

# 문제 43. ReLU 함수를 생성하시오
def ReLU(x):
    return np.maximum(0, x)

x = np.array([0.3, -2])
print(ReLU(x)) # [ 0.3  0. ]

# 문제 44. RelU 함수를 그래프로 그리시오
x = np.arange(-5, 5, 0.1)
y = ReLU(x)
plt.plot(x, y)
plt.ylim(-1.1, 5.1)
plt.xlim(-6, 6)
plt.show()

# 문제 45. 행렬의 재적을 파이썬으로 구현하시오
x = np.array([[1,2,3], [4,5,6]])
y = np.array([[5,6], [7,8], [9,10]])
print(np.dot(x, y))

# 문제 46. 아래의 행렬곱을 파이썬으로 구현하시오
x = np.array([[5,6], [7,8], [9,10]])
y = np.array([[1], [2]])
print(np.dot(x, y))

# 문제 47. 아래의 그림을 파이썬으로 구현하시오
X = np.array([1,2])
W = np.array([[1,3,5], [2,4,6]])
B = np.array([7,8,9])
Y = (np.dot(X, W)) + B
print(Y)

# 문제 48. 위의 문제에서 구한 입력신호의 가중의 합인 y값이 활성함수인 sigmod함수를 통과하면 어떤값이으로 출력되는지 Z값을 확인하시오
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.array([1,2])
W = np.array([[1,3,5], [2,4,6]])
B = np.array([7,8,9])
A1 = (np.dot(X, W)) + B
Z1 = sigmoid(A1)
print(Z1) # [ 0.99999386  0.99999999  1.        ]

# 문제 49. 아래의 그림을 파이썬으로 구현하시오
# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 항등함수
def identity_function(x):
    return x

X = np.array([4.5, 6.2])
W1 = np.array([[0.1,0.2], [0.3,0.4]])
b1 = np.array([0.7, 0.8])
A1 = np.dot(X, W1) + b1
Z1 = sigmoid(A1)
W2 = np.array([[0.5,0.6], [0.7,0.8]])
A2 = np.dot(Z1, W2) + b1
Z2 = sigmoid(A2)
W3 = np.array([[0.1,0.2], [0.3,0.4]])
A3 = np.dot(Z2, W3) + b1
print(A3) # [ 1.05557225  1.33182904]
Y = identity_function(A3)
print(Y) # [ 1.05557225  1.33182904]

# 문제 50. 소프트맥스 함수를 파이썬으로 구현하시오
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y) # [ 0.01821127  0.24519181  0.73659691]
print(np.sum(y)) # 1.0
# "인공 신경망의 출력값으로 확률 벡터를 얻고 싶을 때 사용한다."

# 문제 51. 입력값을 그대로 출력하는 항등 함수를 파이썬으로 구현하시오
def identity_function(x):
    return x

# 문제 52(마지막 문제). 3층 신경망을 파이썬으로 구현하시오
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

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
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)  # [ 0.31682708  0.69627909]
