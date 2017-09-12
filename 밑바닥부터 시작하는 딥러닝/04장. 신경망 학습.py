import numpy as np

# 4.2.1 평균 제곱 오차
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

t = [0,0,1,0,0,0,0,0,0,0] # 정답은 '2'
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0] # 예상값 '2'
print(mean_squared_error(np.array(y),np.array(t))) # 오차값: 0.0975

y1 = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0] # 예상값 '7'
print(mean_squared_error(np.array(y1), np.array(t))) # 오차값: 0.5975

## 4.2.3 미니배치 학습
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test ) = \
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

## 4.2.4 (배치용) 교차 엔트로피 오차 구현하기
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size

print(cross_entropy_error(np.array(y), np.array(t)))

# 미니배치
import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import pickle

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    return x_train,t_train

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return  y

def init_network():
    with open("d:\python\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    return y

def crossEntropyerror(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta)) / len(y)

train= get_data()

x_train = train[0]
t_train = train[1]

print(x_train.shape)
print(t_train.shape)

train_size = 60000
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size)
print(batch_mask)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

y = predict(init_network(), x_batch)
p = np.argmax(y, axis=1)
print(p)

# 4.3 수치 미분
## 4.3.1 미분
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return ( (f(x+h) - f(x-h)) / (2*h) )

## 4.3.2 수치 미분의 예
def function_1(x):
    return 0.01*x**2 + 0.1*x
import numpy as np
import matplotlib.pylab as plt

x = np.arange(0, 20, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

numerical_diff(function_1, 5) # 0.1999999999990898
numerical_diff(function_1, 10) # 0.2999999999986347

## 4.3.3 편미분
# f(x0, x1) = x0^2 + x1^2
import numpy as np

def function_2(x):
    return np.sum(np.square(x))

# 4.4  기울기
def numericalGradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmpVal = x[idx]
        # f(x+h) 계산
        x[idx] = tmpVal + h
        print(x[idx])
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmpVal - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmpVal # 값 복원
    return grad

print(numericalGradient(function_2, np.array([3.0, 4.0])))

## 4.4.1 경사법(경사 하강법)
def gradientDescent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numericalGradient(f, x)
        x -= lr * grad
    return x

def function_2(x):
    return np.sum(np.square(x))

init_x = np.array([-3.0, 4.0])
print(gradientDescent(function_2, init_x=init_x, lr=0.1, step_num=100))

## 4.4.2 신경망에서의 기울기
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
np.argmax(p) # 최댓값의 인덱스

t = np.array([0, 1, 0]) # 정답 레이블
net.loss(x, t)

def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)

f = lambda w:net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)

# 4.5 학습 알고리즘 구현하기
## 4.5.1 2층 신경망 클래스 구현하기
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weighr_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weighr_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weighr_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x: 입력 데이터 t: 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(y, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x: 입력 데이터, t: 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
net.params['W1'].shape
net.params['b1'].shape
net.params['W2'].shape
net.params['b2'].shape






