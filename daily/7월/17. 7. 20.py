# coding: utf-8
import os
import sys

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 훈련데이터/훈련데이터 라벨, 테스트 데이터/테스트 데이터 라벨
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# 설명: flatten=True라는것은 입력이미지를 평탄하게 1차원 배열로 변환한다.
# normalize: 입력 이미지의 픽셀의 값을 0~1사이로 할 것인지 아니면 원래값인 0~255로 할지 결정하느 함수
# 0~255범위의 각 픽셀의 값을 0.0~1.0사이의 범위로 변환을 하는데 이렇게 특정 범위로 변환 처리하는것을?
# "정규화"라고 하고 신경망의 입력데이터에 특정 변환을 가하는것을 "전처리"라 한다.

img = x_train[1]
label = t_train[1]
print(img)

print(img.shape) # (784,)
img = img.reshape(28, 28) # 형상을 원래 이미지의 크기로 변형
print(img.shape) # (28, 28)

img_show(img)

# 문제 53. x_train의 0번째 요소의 필기체 숫자는 5였다. 그렇다면 x_train의 1번째 요소의 필기체 숫자는 무엇인지 확인하시오
img = x_train[1]
label = t_train[1]
print(label) # 0

# 문제 54. 훈련이미지가 60,000장인것을 확인하시오
print(len(x_train))

# 문제 55(점심시간 문제). 필기체 숫자 9를 출력하시오
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from stu.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

for i in range(len(x_train)):
    img = x_train[i]
    label = t_train[i]
    if label == 9:
        print(img.shape) # (784,)
        img = img.reshape(28, 28) # 형상을 원래 이미지의 크기로 변형
        print(img.shape) # (28, 28)

        img_show(img)

##################################################

# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
# 데이터셋을 읽어올때 인터넷이 연결되어 있는 상태에서 가져와야하는데 이때 시간이 걸린다.
# 가져온 이미지를 로컬에 저장할 때 pickle파일로 생성이 되고 로컬에 저장되어 있으면
# 순식간에 읽을 수 있기때문에 임폴트 해야 한다.
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network
# 설명: sample_weight.pkl 파일안에 학습된 가중치와 바이어스가 계산되어서 들어있다.
# 실경망을 어떻게 구현해서 학습하것인지 추정해보면?
# 입력층 : 784개 (이미지 크기가 28x28 픽셀임으로)
# 출력층 : 10개 (0에서 9로 분류함으로)
# 첫번째 은닉층 : 50개 (임의의 수)
# 두번째 은닉층 : 100개 (임의의 수)

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y
# 예) [0.05 0.05 0.7 0.05 0.05 0.25 0.25 0.25 0.15 0.1]

x, t = get_data()
network = init_network()
accuracy_cnt = 0 # 정확도를 출력해주기 위한 변수
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    print(p)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# 문제 56. 위의 코드를 수정해서 하나의 x[34] 의 데스트 데이터가 신경망이 예측한것과 맞는지 확인하시오
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y
# 예) [0.05 0.05 0.7 0.05 0.05 0.25 0.25 0.25 0.15 0.1]

x, t = get_data()
network = init_network()
accuracy_cnt = 0 # 정확도를 출력해주기 위한 변수
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

y = predict(network, x[34])
p = np.argmax(y)
print('테스트 데이터:', t[34], ', 예측 데이터:', p)

# 배치로 돌리는 코드를 이해하기 위한 사전 파있너 코드의 이해
list(range(0,10)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 문제 57. 아래와 같이 결과를 출력하시오
list(range(0,10,3)) # [0, 3, 6, 9]

# 문제 58. [0, 3, 6, 9] 이중 최대값 원소의 인덱스를 출력하시오
a = list(range(0,10,3))
print(np.argmax(a))

# 문제 59. numpy를 이용하여 리스트의 최대값 원소의 인덱스를 출력하시오
a = list(range(0,20,3))
print(np.argmax(a))

# 문제 60. 아래의 행렬 배열을 생성하고 각 행의 최대값에 해당하는 인덱스가 출력되게하시오
a = np.array([[0.1,0.8,0.1], [0.3,0.1,0.6], [0.2,0.5,0.3], [0.8,0.1,0.1]])
for i in range(len(a)):
    print(np.argmax(a[i]))

x = np.array([[0.1,0.8,0.1], [0.3,0.1,0.6], [0.2,0.5,0.3], [0.8,0.1,0.1]])
# 행의 최대값 인덱스
y = np.argmax(x, axis=1)
print(y)

# 열의 최대값 인덱스
y = np.argmax(x, axis=0)
print(y)

# 문제 61. 아래 2개의 리스트를 만들고 서로 같은 숫자가 몇 개가 있는지 출력하시오
X = np.array([2,1,3,5,1,4,2,1,1,0])
Y = np.array([2,1,3,4,5,4,2,1,1,2])
print(X == Y)
print(np.sum(X == Y))

# 문제 62. 아래의 리스트를 x라는 변수에 담고 앞의 5개의 숫자만 출력하시오
x = list(range(1, 11))
print(x[:5])

# 문제 63. 100장의 이미지를 한번에 입력층에 넣어서 추론하는 신경망 코드를 수행하시오
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("d:\\data\\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


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


x, t = get_data()
network = init_network()

batch_size = 1 # 배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    # print(i)
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# 문제 64. batch_size를 1로 했을 때 100으로 했을 때 수행속도의 차이가 있는지 확인하시오
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
import time

start_t = time.time()
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("d:\\data\\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


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


x, t = get_data()
network = init_network()

batch_size = 1 # 배치 크기
accuracy_cnt = 0

# print(x[0]) # 필기체 숫자 5를 출력하는 픽셀 784개
# print(x[0:2]) # 필기체 숫자 2개가의 픽셀이 출력된다.
# print(x[0:100]) # 인덱스 99에 해당하는 필기체까지의 픽셀이 출력

for i in range(0, len(x), batch_size):
    # print(i)
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
end_t = time.time()
print(end_t - start_t)

# 문제 65. 훈련 데이터로 batch_size 1로 했을때와 batch_size 100으로 했을때의 정확도의 수행속도의 차이를 비교하시오
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
import time

start_t = time.time()
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_train, t_train


def init_network():
    with open("d:\\data\\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


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


x, t = get_data()
network = init_network()

batch_size = 100 # 배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    # print(i)
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
end_t = time.time()
print(end_t - start_t)

