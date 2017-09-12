# 문제 154. 0부터 15까지 원소로 이루어진 4x4행렬을 만드시오
import numpy as np
A = np.array([i for i in range(16)]).reshape(4, 4)
print(A)

# 문제 155. 위에서 만든 행렬에 제로패딩 1을 수행하시오
B = np.pad(A, pad_width=1, mode='constant', constant_values=0)
print(B)

# 문제 156. 0부터 15까지 원소로 이루어진 4x4행렬을 만들고 0부터 8까지의 원소로 이루어진 3x3필터를 이용해서 합성곱을 하시오(스트라이드=1)
import numpy as np
input = np.array([i for i in range(16)]).reshape(4,4)
Filter = np.array([i for i in range(9)]).reshape(3,3)

result = []
for i in range(len(input) -2):
    for j in range(len(input) -2):
        result.append(np.sum(input[i:i+3, j:j+3] * Filter))

result = np.array(result).reshape(2,2)
print(result)

# 문제 157. 0~35원소 6x6행렬, 0~15까지 4x4필터를 이용해서 합서곱하시오
import numpy as np
input = np.array([i for i in range(36)]).reshape(6,6)
Filter = np.array([i for i in range(16)]).reshape(4,4)

result = []
for i in range(len(input) -3):
    for j in range(len(input) -3):
        result.append(np.sum(input[i:i+4, j:j+4] * Filter))
print(result)

result = np.array(result).reshape(3,3)
print(result)
# 3x3으로 결과가 출력되는데, 가장자리는 출력되지 않았다.
# 위 결과를 입력 shape인 6x6행렬로 출력되게 하려면 어떻게 해야하는가?

# 문제 158. 아래와 같이 출력값(OH)와 Straid(S)와 입력값(H)와 필터값(FH)을 입력하면 패딩(P)가 출력되는 함수를 생성하시오
padding = lambda OH,S,H,FH : ((OH - 1) * S -H + FH) / 2
print(padding(6,1,6,4))

# 문제 159. 0~15의 4x4행렬을 만들고, 0~8의 3x3 필터를 이용한 합성곱(단, S=1, Out 4x4가 되도록 패딩을 적용하시오)
import numpy as np
input = np.array([i for i in range(16)]).reshape(4,4)
Filter = np.array([i for i in range(9)]).reshape(3,3)
print(padding(4,1,4,3))
input_pad = np.pad(input, pad_width=1, mode='constant', constant_values=0)
print(input_pad)

result = []
for i in range(len(input_pad) -2):
    for j in range(len(input_pad) -2):
        result.append(np.sum(input_pad[i:i+3, j:j+3] * Filter))
print(result)

result = np.array(result).reshape(4,4)
print(result)

# 문제 160. 6x6행렬, 3x3필터 합성곱(단, S=1, 출력행렬 6x6)
import numpy as np
input = np.array([i for i in range(36)]).reshape(6,6)
Filter = np.array([i for i in range(9)]).reshape(3,3)
print(padding(6,1,6,3))
input_pad = np.pad(input, pad_width=1, mode='constant', constant_values=0)
print(input_pad)

result = []
for i in range(len(input_pad) -2):
    for j in range(len(input_pad) -2):
        result.append(np.sum(input_pad[i:i+3, j:j+3] * Filter))
print(result)

result = np.array(result).reshape(6,6)
print(result)

# 문제 161. 0~24 5x5행렬, 0~3 2x2필터의 합성곱(단, S=1, 출력행렬 5x5)
import numpy as np
input = np.array([i for i in range(25)]).reshape(5,5)
Filter = np.array([i for i in range(4)]).reshape(2,2)
print(padding(5,1,5,2))
input_pad = np.pad(input, pad_width=((1,0), (1,0)), mode='constant', constant_values=0)
# 패드는 위(1), 아래(0), 왼쪽(1), 오른쪽(0)으로 설정한다.
print(input_pad)
len(input_pad)
result = []
for i in range(len(input_pad) -1):
    for j in range(len(input_pad) -1):
        result.append(np.sum(input_pad[i:i+2, j:j+2] * Filter))
print(result)

print(len(result))
result = np.array(result).reshape(5,5)
print(result)

import numpy as np
x = np.array([ [[1,2,0,0], [0,1,-2,0], [0,0,1,2], [2,0,0,1]], [[1,0,0,0], [0,0,-2,-1], [3,0,1,0], [2,0,0,1]] ])
f = np.array([ [[-1,0,3], [2,0,-1], [0,2,1]], [[0,0,0], [2,0,-1], [0,-2,1]] ])
print(x.shape[0])

result = []
for c in range(x.shape[0]):
    array = []
    for i in range(2):
        temp = []
        for j in range(2):
            temp.append(np.sum(x[c, i:i+3, j:j+3] * f[c]))
        array.append(temp)
    result.append(array)

a1 = np.sum(np.array(result), axis=0)
print(a1)
a2 = (a1.T + [0,2]).T
print(a2)

# 문제 163. 입력 데이터를 15x8 0~199번 원소로 생성해서 채널 10개 입력 데이터, 필터 3x3(0~8까지 원소) 채널 10개 합성곱을 구현하시오
padding = lambda OH,S,H,FH : ((OH - 1) * S -H + FH) / 2
print(padding(15,1,15,3))

x = np.array([i for i in range(120)]*10).reshape(10,15,8)
print(len(x))
xp = np.pad(x, pad_width=((0,0),(1,1),(1,1)), mode='constant', constant_values=0)
len(xp)
f = np.array([i for i in range(9)]*10).reshape(10,3,3)

# result = np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])
result = np.array([0,0,0,0,0,0,0,0]*15).reshape(1,15,8)
result1 = []
for c in range(xp.shape[0]):
    for i in range(xp.shape[1]-2):
        for j in range(xp.shape[2]-2):
            result1.append(np.sum(xp[c, i:i+3, j:j+3] * f[c]))
    result += np.array(result1).reshape(15,8)
    result1 = []

result = np.array(result).reshape(1,15,8)
print(result)

