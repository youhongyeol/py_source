import numpy as np

# 평균 제곱 오차
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

t = [0,0,1,0,0,0,0,0,0,0] # '2'
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0] # 예상 '2'
y2 = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0] # 예상 '7'

# 문제 66. 위의 one-hot encoding된 t값과 확률 y,y2를 각각 평균 제곱 오차함수를 출력하시오!
print(mean_squared_error(np.array(y), np.array(t))) # 0.0975
print(mean_squared_error(np.array(y2), np.array(t))) # 0.5975
# y의 오차가 작으므로 y가 정답에 가깝다고 볼 수 있다.

# 교차 엔트로피 오차
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

 # '2'
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0] # 예상 '2'
y2 = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0] # 예상 '7'
print(cross_entropy_error(np.array(y), np.array(t))) # 0.510825457099
print(cross_entropy_error(np.array(y2), np.array(t))) # 2.30258409299
# 설명: t와의 오차가 평균 제곱 오차보다 더 큰것으로 확인되고 있다.

# 문제 67(점심시간 문제). 아래의 numpy배열(확률)을 교차 엔트로피 오차함수를 이용하여 오차율을 for loop문을 사용해서 확인하시오!
t = [0,0,1,0,0,0,0,0,0,0] # '2'
y1 = [0.6,0.1,0.05,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
y2 = [0.1,0.6,0.05,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
y3 = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0] # '2'
y4 = [0.1,0.05,0.0,0.6,0.05,0.1,0.0,0.1,0.0,0.0]
y5 = [0.1,0.05,0.0,0.05,0.6,0.1,0.0,0.1,0.0,0.0]
y6 = [0.1,0.05,0.0,0.05,0.1,0.6,0.0,0.1,0.0,0.0]
y7 = [0.1,0.05,0.0,0.05,0.1,0.0,0.6,0.1,0.0,0.0]
y8 = [0.1,0.05,0.0,0.05,0.1,0.0,0.1,0.6,0.0,0.0]
y9 = [0.1,0.05,0.0,0.05,0.1,0.0,0.1,0.0,0.6,0.0]

for i in range(1,10):
    print ("y"+str(i), " 의 오차율")
    print(cross_entropy_error(np.array(eval('y'+str(i))), np.array(t)))


for i in range(1,10):
    y = []
    y.append(np.array(eval("y"+str(i))))

print(y)

# 문제 68. 60,000 미만의 숫자중 무작위로 10개를 출력하시오
print(np.random.choice(60000, 10))

# 문제 69. mnist에서 np.random.choice()를 이용하여 60,000중 10개의 데이터를 출력
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

print(x_train.shape)
print(t_train.shape)

train_size = 60000 # (60000, 784)
batch_size = 10 # (60000, )
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(len(x_batch)) # 10
print(x_batch.shape) # (10, 784)

# 문제 70. 데이터 1개를 가지고 오차를 구하는 교차 엔트로피 오차 함수는 아래와 같이 만들었다. 그렇다면 배치용 교차 엔트로피 오차는 어떻게 생성해야 하는가?
import numpy as np

def cross_entropy_error1(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta)) / len(y)

# 문제 71. 교차 엔트로피 오차를 사용 코드를 구현하시오!
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
    with open("D:\python\sample_weight.pkl", 'rb') as f:
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

train = get_data()

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
print('CEE: ', crossEntropyerror(y,t_batch))

print(np.float32(1e-50))

# 문제 72. 근사로 구한 미분 함수를 파이썬으로 구현하시오!(h = 0.0001)
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

# 문제 73. y =0.01 * x^2 + 0.1 * x 함수를 미분하는데 x가 10일때 미분계수는 어떻게 되는가?
def function_1(x):
    return 0.01 * x^2 + 0.1 * x
