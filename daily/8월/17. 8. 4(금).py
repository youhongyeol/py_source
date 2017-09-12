padding = lambda OH,S,H,FH : ((OH - 1) * S -H + FH) / 2
print(padding(6,1,6,4))

# 문제3. 0~80원소를 가진 9x9행렬, 0~15원소를 가진 4x4행렬(단, S=1, 출력행렬=9x9)
import numpy as np
x = np.array([i for i in range(81)]).reshape(9,9)
f = np.array([i for i in range(16)]).reshape(4,4)
print(padding(9,1,9,4))
xp = np.pad(x, pad_width=((2,1),(2,1)), mode='constant', constant_values=0)
print(xp)

result = []
for i in range(len(xp) -3):
    for j in range(len(xp) -3):
        result.append(np.sum(xp[i:i+4, j:j+4] * f))

result = np.array(result).reshape(9,9)
print(result)

# 문제4. 위와 같으나 채널 수가 3개일 때 합성곱을 구하시오
x = np.array([i for i in range(81)]*3).reshape(3,9,9)
f = np.array([i for i in range(16)]*3).reshape(3,4,4)
xp = np.pad(x, pad_width=((0,0),(2,1),(2,1)), mode='constant', constant_values=0)
result = np.array([0,0,0,0,0,0,0,0,0]*9).reshape(1,9,9)
result1 = []
for c in range(xp.shape[0]):
    for i in range(xp.shape[1]-3):
        for j in range(xp.shape[2]-3):
            result1.append(np.sum(xp[c, i:i+4, j:j+4] * f[c]))
    result += np.array(result1).reshape(9,9)
    result1 = []

result = np.array(result).reshape(1,9,9)
print(result)

# 문제 162. 아래의 이미지를 max pooling 하시오
import numpy as np
x = np.array([[21,8,8,12], [12,19,9,7], [8,10,4,3], [18,12,9,10]])

# 문제 163. 아래의 이미지를 평균 풀링 하시오
x = np.array([[21,8,8,12], [12,19,9,7], [8,10,4,3], [18,12,9,10]])

# 문제 164(점심시간 문제). 아래의 행렬의 최대 풀링을 파이썬으로 구현하시오
x = np.array([[21,8,8,12], [12,19,9,7], [8,10,4,3], [18,12,9,10]])

def Maxpooling(x):
    result = []
    for i in range(0, len(x), 2):
        for j in range(0, len(x[0]), 2):
            tmp = []
            # print(i)
            # print(j)
            tmp.append(x[i:i+2, j:j+2])
            # print(tmp)
            a = np.max(tmp)
            # print(a)
            result.append(a)
            # print(result)
    result = np.array(result).reshape(2,2)
    return result

print(Maxpooling(x))

# 문제 165. 풀링의 이해
import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).

    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.

    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None
        self.Mean = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # arg_max = np.argmax(col, axis=1)
        out = np.mean(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

data = np.array(
       [[
         [[1, 2, 3, 0],
          [0, 1, 2, 4],
          [1, 0, 4, 2],
          [3, 2, 0, 1]],
         [[3, 0, 6, 5],
          [4, 2, 4, 3],
          [3, 0, 1, 0],
          [2, 3, 3, 1]],
         [[4, 2, 1, 2],
          [0, 1, 0, 4],
          [3, 0, 6, 2],
          [4, 2, 4, 5]]
       ]])
max_pool = Pooling(2, 2)
forward_max = max_pool.forward(data)
print(data.shape)
print(forward_max.shape)
print(data)
print(forward_max)

# 문제 166. 아래와 같은 4차원 데이터를 생성하고 5x5의 가중치로 필터링 된 데이터와 합성곱을 하기 편하게 아래 4차원 데이터를 im2col을 이용해서 2차원으로 변경하시오
import sys, os
sys.path.append(os.pardir)
from common.util import im2col
import numpy as np

x1 = np.random.rand(1,3,7,7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape) # (9, 75)

x2 = np.random.rand(10,3,7,7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape) # (90, 75)

# 문제 167. 0~15원소 4x4행렬, 0~8원소 3x3 필터(단, S=1 im2col 활용)
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, H, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out


##################################################
##################################################

import sys, os
sys.path.append(os.pardir)
import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).

    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride * out_h  # y=0일 때 y_max=2,  y=1일 때 y_max=3,  y=2일 때 y_max=4
        for x in range(filter_w):
            x_max = x + stride * out_w  # x=0일 때 x_max=2,  x=1일 때 x_max=3,  x=2일 때 x_max=4
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
            # 디버깅 variables화면/ col 우측 클릭 => view as Array 클릭

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    # col.shape = (N,C,filter_h,filter_w,out_h,out_w)  =>  col.shape(N,out_h,out_w,C,filter_h,filter_w) 로 transpose 이후
    # (N*out_h*out_w, C*filter_h*filter_w) 의 2차원 행렬로 reshape
    # print(col.shape)
    return col

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)  # col.shape = (N*out_h*out_w, C*FH*FH)
        # print(col.shape)
        col_W = self.W.reshape(FN, -1).T  # (FN,C,FH,FW)  reshape=> (FN, C*FH*FW)  transpose=> (C*FH*FW, FN)
        out = np.dot(col, col_W) + self.b  # 결과 차원 (N*out_h*out_w,FN)

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # out.reshape의 shape = (N,out_h,out_w,FN)
        # transpose이후 shape = (N,FN,out_h,out_w)
        return out


x = np.arange(48).reshape(1, 3, 4, 4)
x
W = np.arange(54).reshape(2, 3, 3, 3)
W
b = 1
conv = Convolution(W, b)
f = conv.forward(x)
print('f = ', f, 'f.shape = ', f.shape)   # N, FN, out_h, out_w


# 문제1  x1,  W1, b1이 다음과 같을 때 convolution 계층을 거친 뒤 feature map의 차원은?
x1 = np.arange(192).reshape(1, 3, 8, 8)
W1 = np.arange(135).reshape(5, 3, 3, 3)
b1 = 1
conv1 = Convolution(W1, b1)
f1 = conv1.forward(x1)
print('f1 = ', f1, 'f1.shape = ', f1.shape)  # N, FN, out_h, out_w

# 문제2. x1, W1, b1이 다음과 같을 때 convolution 계층을 거친 뒤 나오는 feature map의 차원은?
x1 = np.arange(64).reshape(1, 1, 8, 8)
W1 = np.arange(45).reshape(5, 1, 3, 3)
b1 = 1
conv2 = Convolution(W1, b1, stride=2, pad=1)
f2 = conv2.forward(x1)
print('f2 = ', f2, 'f2 shape = ', f2.shape)

import numpy as np

#실행시 exercise마다 """ 주석 제거하고 실행

###################################################################### EX01 np.pad
"""
a=np.ones((4,3,2))
print('a = ',a)
npad=[(0,0),(1,3),(2,1)]


b=np.pad(a,pad_width=npad,mode='constant',constant_values=1)  
print('b.shape = ',b.shape)
print('b = ',b)
"""
####################################################################### EX02 a.transpose()
"""
a=np.array([[[1,2,3,7],[3,4,5,8],[0,1,1,9]],[[5,6,7,8],[7,8,9,10],[0,2,1,5]]])
print(a,a.shape)   
#[[[ 1  2  3  7]
#  [ 3  4  5  8]
#  [ 0  1  1  9]]
#
# [[ 5  6  7  8]
#  [ 7  8  9 10]
#  [ 0  2  1  5]]]
# (2,3,4)


a_T=a.transpose(2,0,1)    # (2,3,4) => (4,2,3)   즉 행렬 a 안의 (1,2,3) 요소가 (3,1,2) 위치로! 
print(a_T,a_T.shape)
#[[[ 1  3  0]
#  [ 5  7  0]]
#
# [[ 2  4  1]
#  [ 6  8  2]]
#
# [[ 3  5  1]
#  [ 7  9  1]]
#
# [[ 7  8  9]
#  [ 8 10  5]]]
#(4, 2, 3)

a_T=a.transpose(2,1,0)
a_reshape=a.reshape([4,3,2])     # reshape와의 차이 확인 => reshape는 단순히 나열된 순서대로 요소를 가져와 shape에 맞게 재배열할 뿐
print(a_T,a_T.shape)
print(a_reshape.shape,a_reshape)
#[[[ 1  5]
#  [ 3  7]
#  [ 0  0]]
#
# [[ 2  6]
#  [ 4  8]
#  [ 1  2]]
#
# [[ 3  7]
#  [ 5  9]
#  [ 1  1]]
#
# [[ 7  8]
#  [ 8 10]
#  [ 9  5]]]
# (4, 3, 2)
"""