## 1.3.4 리스트
a = [1, 2, 3, 4, 5] # 리스트 생성
print(a) # 리스트의 내용 출력
print(len(a)) # 리스트의 길이 출력
a[0] # 첫 원소에 접근
a[4] # 다섯 번째 원소에 접근
a[4] = 99 # 값 대입
print(a)

# 슬라이싱
a[0:2] # 인덱스 0부터 2까지 얻기(2번째는 포함하지 않는다!)
a[1:] # 인덱스 1부터 끝까지 얻기
a[:3] # 처음부터 인덱스 3까지 얻기(3번째는 포함하지 않는다!)
a[:-1] # 처음부터 마지막 원소의 1개 앞까지 얻기
a[:-2] # 처음부터 마지막 원소의 2개 앞까지 얻기

## 1.3.5 딕셔너리
me = {'height':180} # 딕셔너리 생성
me['height'] # 원소에 접근
me['weight'] = 70 # 새 원소 추가
print(me)

## 1.3.6 bool(True와 False)
hungry = True # 배가 고프다.
sleepy = False # 졸리지 않다.
type(hungry)
not hungry
hungry and sleepy # 배가 고프다 그리고 졸리지 않다. False
hungry or sleepy # 배가 고프다 또는 졸리지 않다. True

## 1.3.7 if 문
hungry = True
if hungry:
    print("I'm hungry")
hungry = False
if hungry:
    print("I'm hungry")
else:
    print("I'm not hungry")
    print("I'm sleepy")


## 1.3.8 for 문
for i in[1, 2, 3]:
    print(i)

## 1.3.9 함수
def hello():
    print("Hello World!!")
hello()

def hello(object):
    print("Hello " + object + "!!")
hello("dog")

## 1.4.2 클래스
class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized!")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good-bye " + self.name + "!")

m = Man("Lion")
m.hello()
m.goodbye()

# 1.5 넘파이
## 1.5.2 넘파이 배열 생성하기
import numpy as np
x = np.array([1.0, 2.0, 3.0])
print(x)
type(x)

## 1.5.3 넘파이의 산수 연산
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
x + y # 원소별 덧셈
x - y
x * y # 원소별 곱셈
x / y

## 1.5.4 넘파이의 N차원 배열
A = np.array([ [1,2], [3,4] ])
print(A)
A.shape # 행열의 형상을 나타낸다.
A.dtype
B = np.array([ [3,0], [0,6]])
A + B
A * B
A * 10

## 1.5.5 브로드캐스트
A = np.array([ [1,2], [3,4] ])
B = np.array([ [10,20] ])
A * B

## 1.5.6 원소 접근
X = np.array([ [51,55], [14,19], [0,4] ])
print(X)
X[0] # 0행
X[0][1] # (0, 1) 위치의 원소

for row in X: # for 문으로 원소에 접근
    print(row)

X = X.flatten() # X를 1차원 배열로 변환(평탄화)
print(X)
X[np.array([0, 2, 4])] # 인덱스가 0, 2, 4인 원소 얻기

X > 15 # array([ True,  True, False,  True, False, False], dtype=bool)
X[X>15] # array([51, 55, 19])

# 1.6 matplotlib
## 1.6.1 단순한 그래프 그리기
import numpy as np
import  matplotlib.pyplot as plt

# 데이터 준비
x = np.arange(0, 6, 0.1) # 0에서 6까지 0.1 간격으로 생성
y = np.sin(x)

# 그래프 그리기
plt.plot(x, y)

## 1.6.2 pyplot의 기능
import numpy as np
import  matplotlib.pyplot as plt

# 데이터 준비
x = np.arange(0, 6, 0.1) # 0에서 6까지 0.1 간격으로 생성
y1 = np.sin(x)
y2 = np.cos(x)

# 그래프 그리기
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos") # cos 함수는 점선으로 그리기
plt.xlabel("x") # x축 이름
plt.xlabel("y") # y축 이름
plt.title('sin & cos') # 제목
plt.legend()
plt.show()

## 1.6.3 이미지 표시하기
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread("D:\\data\\태극기.jpg") # 이미지 읽어오기(적절한 경로를 설정하세요!!)

plt.imshow(img)