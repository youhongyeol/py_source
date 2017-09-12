# 문제 110. 아래의 행렬식을 손으로 구현하시오
# 문제 111. 행렬식을 numpy를 이용해서 구현하시오
import numpy as np
x = np.array([[1,2,3], [4,5,6]])
y = np.array([[1,2], [3,4], [5,6]])
print(np.dot(x,y))

# 문제 112. 아래의 행렬식을 손으로 구현하시오
x = np.array([[1,2], [3,4], [5,6]])
y = np.array([7,8])
print(y.shape)
print(np.dot(x,y))

# 문제 113. 아래의 행렬식을 손으로 구현하시오
x = np.array([[1,2], [3,4], [5,6]])
y = np.array([[1,2,3,4], [5,6,7,8]])
print(np.dot(x,y))

# 문제 114. 위의 행렬식을 파이썬으로 구현하시오
x = np.array([[1,2], [3,4], [5,6]])
y = np.array([[1,2,3,4], [5,6,7,8]])
print(np.dot(x,y))

# 문제 115. 아래의 행렬식을 손으로 구현하시오
x = np.array([1,2])
y = np.array([[1,3,5], [2,4,6]])
print(np.dot(x,y))
# 차원의 원소수가 동일해야 한다.

# 문제 116. (단층)아래의 신경망을 행렬의 내적으로 구현해서 출력값 y를 출력하시오
x = np.array([1,2])
y = np.array([[1,3,5], [2,4,6]])
print(np.dot(x,y))

# 문제 117. 2층 신경망
X = np.array([1,2])
W1 = np.array([[1,3,5], [2,4,6]])
A1 = np.dot(X,W1)
print(A1)
W2 = np.array([[1,4], [2,5], [3,6]])
A2 = np.dot(A1, W2)
print(A2)

# 문제 118. 3층 신경망 구현
X = np.array([1,2])
W1 = np.array([[1,3,5], [2,4,6]])
b1 = np.array([1,2,3])
A1 = np.dot(X, W1) + b1
W2 = np.array([[1,4], [2,5], [3,6]])
b2 = np.array([1,2])
A2 = np.dot(A1, W2) + b2
W3 = np.array([[1,3], [2,4]])
b3 = np.array([1,2])
A3 = np.dot(A2, W3) + b3
print(A3)

# 문제 119. 신경망의 역전파를 구현하시오

# 문제 120. 순전파, 역전파를 구하시오
X = np.array([1,2])
W = np.array([[1,3,5], [2,4,6]])
b = np.array([1,2,3])

# 문제 121. 순전파를 파이썬으로 구현하시오
X = np.array([1,2])
W = np.array([[1,3,5], [2,4,6]])
b = np.array([1,2,3])

def forward(x, w, b):
    return np.dot(x,w) + b

print(forward(X,W,b))

# 문제 122. 역전파 함수를 파이썬으로 구현하시오
import numpy as np
out = np.array([6,13,20], ndmin=2)
x = np.array([1,2], ndmin=2)
w = np.array([[1,3,5], [2,4,6]])
b = np.array([1,2,3])

def backward(x,w,out):
    dx = np.dot(out, w.T)
    dw = np.dot(x.T, out)
    db = np.sum(out, axis=0)
    return dx, dw, db

print(backward(x,w,out))

# 문제 122. 위에서 만든 forward, backward 함수를 class Affine이라 생성하시오
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, x, dout):
        dx = np.dot(dout, self.W.T)
        dw = np.dot(self.x.T, dout)
        db = dout
        return dx, dw, db

# 문제 123. 아래의 2층 신경망의 순전파를 Affine 클래스를 사용해서 출력하시오
x = np.array([1,2], ndmin=2)
w1 = np.array([[1,3,5], [2,4,6]])
b1 = np.array([1,2,3])
w2 = np.array([[1,4], [2,5], [3,6]])
b2 = np.array([1,2])

affine1 = Affine(w1, b1)
affine2 = Affine(w2, b2)
out = affine1.forward(x)
out2 = affine2.forward(out)
print(out2)

# 문제 124. 2층 긴경망의 역전파를 Affine 클래스를 사용해서 구현하시오
dx2, dw2, db2 = affine2.backward(out, out2)
dx1, dw1, db1 = affine1.backward(x, dx2)
print('dx1\n', dx1, '\ndw1\n', dw1, '\ndb1\n', db1)

# 문제 125. 2층 신경망의 순전파를 구현하는데 은닉층에 활성함수 ReLU 함수를 추가하여 구현하시오
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

x = np.array([1,2], ndmin=2)
w1 = np.array([[1,3,5], [2,4,6]])
b1 = np.array([1,2,3])
w2 = np.array([[1,4], [2,5], [3,6]])
b2 = np.array([1,2])

affine1 = Affine(w1, b1)
affine2 = Affine(w2, b2)
relu1 = ReLU()

A1 = affine1.forward(x)
Z1 = relu1.forward(A1)
A2 = affine2.forward(Z1)
print(A2)

# 위의 코드의 역전파를 구현하시오
dx2, dw2, db2 = affine2.backward(Z1, A2)
dx1 = relu1.backward(dx2)
dx, dw, db = affine1.backward(x, dx1)
print('dx\n',dx, '\ndw\n', dw, '\ndb\n', db)

