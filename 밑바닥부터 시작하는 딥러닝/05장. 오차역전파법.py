# 5.4 단순한 계층 구현하기
## 5.4.1 곱셈 계층
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

apple = 100
apple_num = 2
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price)

# 역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dtax)

## 5.4.2 덧셈 계층
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

apple = 100
apple_n = 2
citrus = 150
citrus_n = 3

# 계층
mul_apple_p = MulLayer()
mul_citrus_p = MulLayer()
add_apple_citrus_p = AddLayer()
mul_tax_p = MulLayer()

# 순전파
apple_p = mul_apple_p.forward(apple, apple_n)
citrus_p = mul_citrus_p.forward(citrus, citrus_n)
add = add_apple_citrus_p.forward(apple_p, citrus_p)
price = mul_tax_p.forward(add, tax)
print(price) # 715

# 역전파
dprice = 1
dadd, dtax = mul_tax_p.backward(dprice)
dapple_p, dcitrus_p = add_apple_citrus_p.backward(dadd)
dcitrus, dcitrus_n = mul_citrus_p.backward(dcitrus_p)
dapple, dapple_n = mul_apple_p.backward(dapple_p)
print(dapple, dapple_n, dcitrus, dcitrus_n, dtax)

import numpy as np
import copy

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + enp.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

X = np.random.rand(2) # 입력
W = np.random.rand(2,3) # 가중치
b = np.random.rand(3) # 편향

Y = np.dot(X,W) + b
print(Y)

import numpy as np

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx, self.dW, self.db

x = np.array([1,2], ndmin=2)
w1 = np.array([[1,3,5], [2,4,6]])
b1 = np.array([1,2,3])
w2 = np.array([[1,4], [2,5], [3,6]])
b2 = np.array([1,2])

affine1 = Affine(w1, b1)
affine2 = Affine(w2, b2)

# 순전파
A1 = affine1.forward(x)
print(A1)
A2 = affine2.forward(A1)
print(A2)

# 역전파
dx2, dw2, db2 = affine2.backward(A2)
print(dx2, dw2, db2)
dx, dw, db = affine1.backward(dx2)
print('dw\n', dx, '\ndw\n', dw, '\ndw\n', db)