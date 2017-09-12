# 문제 88. 2x3의 가중치를 랜덤으로 생성하고 간단한 신경망을 구현해서 기울기를 구하는 파이썬 코드를 작성하시오
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simplNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 2x3의 가중치 배열을 랜덤으로 생성

    def predict(self, x): # 행렬 곱
        return np.dot(x, self.W)

    def loss(self, x, t): # 손실함수
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

# 문제 89. 위에서 구현한 신경망에 입력값[0.6,0.9]를 입력하고 target은 [0,0,1]로 오차가 얼마나 발생하는지 확인하시오
x = np.array([0.6,0.9])
t = np.array([0,0,1])

net = simplNet()
print(net.W)
p = net.predict(x)
np.argmax(p) # 최댓값 인덱스
print(p)
print(net.loss(x,t))

# 문제 90. 수치미분함수에 위에서 만든 신경망의 비용함수와 가중치(2x3)의 가중치를 입력해서 기울기(2x3)를 구하시오
def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)
# print(net.W)
# [[-1.43750239 -0.15283551  0.15229588]
#  [ 0.1294804  -0.59152713  0.87976991]]
# print(dW)
# [[ 0.08299767  0.09375696 -0.17675463]
#  [ 0.12449651  0.14063544 -0.26513195]]

# 문제 91(점심시간 문제). 아래에서 만든 함수 f를 lambda식으로 구현해서 f라는 변수에 넣고 수행하면 기울기가 출력되게 하시오
f1 = lambda W:net.loss(x, t)

dW = numerical_gradient(f1, net.W)
print(dW)

# 2층 신경망 코드 구현하기
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1 # 100x784 * 784x50 + 50 = 100x50
        z1 = sigmoid(a1) # 100x50
        a2 = np.dot(z1, W2) + b2 # 100x50 * 50x10 + 50 = 100x10
        y = softmax(a2) # 100x10
        return y

    def loss(self, x,t): # 오차를 구하는 함수
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t): # 정확도 출력
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t): # 수치미분
        loss_W = lambda W: self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) # 784x 50의 기울기
        grads['b1'] = numerical_gradient(loss_W, self.params['b1']) # 50개의 bias
        grads['W2'] = numerical_gradient(loss_W, self.params['W2']) # 50x10의 기울기
        grads['b2'] = numerical_gradient(loss_W, self.params['b2']) # 10개의 bias
        return grads

# 문제 93. b1의 배열을 확인하시오
b1 = np.zeros(50)
print(b1)

# 문제 94. 아래의 x(입력값), t(target 값), y(예상값)을 아래와 같이 설정하고 위에서 만든 2층 신경망을 객체화해서 W1, W2, b1, b2의 차원이 어떻게 되는지 출력하시오
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
x = np.random.rand(100, 784)
x.shape # (100, 784)
y = net.predict(x)
y.shape # (100. 10)
t = np.random.rand(100,10)
t.shape # (100, 10)

print(net.params['W1'].shape) # (784, 100)
print(net.params['b1'].shape) # (100,)
print(net.params['W2'].shape) # (100, 10)
print(net.params['b2'].shape) # (10,)

# 문제 95. x(입력값), t(target 값), y(예상값)을 아래와 같이 설정하고 위에서 만든 2층 신경망을 객체화해서 W1, W2, b1, b2의 기울기의 차원이 어떻게 되는지 출력하시오
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

x = np.random.rand(1,784)
y = net.predict(x)
t = np.random.rand(1,10)

grads = net.numerical_gradient(x, t)
print(grads['W1'].shape) # (784, 100)
print(grads['b1'].shape) # (100,)
print(grads['W2'].shape) # (100, 10)
print(grads['b2'].shape) # (10,)

##미니 배치
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# 60000x784, 60000x10, 10000x784, 10000x10
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0] # 60000 개
print(x_train.shape[1])  # 784개
batch_size = 100  # 미니배치 크기
learning_rate = 0.1
train_loss_list = []
train_acc_list = [] # 훈련 데이터 정확도
test_acc_list = []  # 테스트 데이터 정확도
# 훈련과 데스트 데이터를 같이 한번에 정확도를 계산하려고 빈 리스트를 만들고 있는데 왜? 두개를 같이 확인하냐면 훈련 데이터가 혹시 오버피팅 되지는 앉았는지 확인하려고

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
print('1에폭: ',iter_per_epoch) # 600

for i in range(iters_num): # 10000
    # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
    batch_mask = np.random.choice(train_size, batch_size) # 100개 씩 뽑아서 10000번 백만번
    x_batch = x_train[batch_mask]
    # print(x_batch.shape) #100 x 784
    t_batch = t_train[batch_mask]
    # print(t_batch.shape) # 100 x 10

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch) # 성능 개선판!

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss) # cost 가 점점 줄어드는것을 보려고
    # 1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을때 정확도만 체크

    if i % iter_per_epoch == 0: # 600 번마다 정확도 쌓는다.
        # print(x_train.shape) # (60000,784)
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc) # 10000/600 개  16개 # 정확도가 점점 올라감
        test_acc_list.append(test_acc)  # 10000/600 개 16개 # 정확도가 점점 올라감
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# 문제 96. 10000번 돌릴때 결과를 출력하시오

# 문제 97(마지막 문제). numerical_gradient함수 말고 5장에서 배울 오차 역전파(gradient)함수를 사용해서 정확도를 계산하시오
# 1에폭:  600.0
# train acc, test acc | 0.0993, 0.1032
# train acc, test acc | 0.944233333333, 0.94
# train acc, test acc | 0.9458, 0.9427
