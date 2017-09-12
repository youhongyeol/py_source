#예제1: 배열 만들기
import numpy as np

a = np.array([[1,2], [3,4]])
print(a)

# 예제2: 사칙연산
# a + n : 배열의 모든 원소에 n만큼을 더한다.
# a - n : 배열의 모든 원소에 n만큼을 뺀다.
# a * n : 배열의 모든 원소에 n만큼을 곱한다.
# a / n : 배열의 모든 원소에 n만큼을 나눈다.

# 문제 1. 아래의 a배열에 모든 원소에 5를 더한 결과를 출력하시오
a = np.array([[1,2], [3,4]])
print(a + 5)

# 문제 2. 아래의 배열의 원소들의 평균값을 출력하시오
a = np.array([1,2,4,5,5,7,10,13,18,21])
print(np.mean(a))

# 문제 3. a배열의 중앙값을 출력하시오
a = np.array([1,2,4,5,5,7,10,13,18,21])
print(np.median(a))

# 문제 4. a배열의 최대값과 최소값을 출력하시오
a = np.array([1,2,4,5,5,7,10,13,18,21])
print('최대값: ', np.max(a), '최소값: ', np.min(a))

# 문제 5. a배열의 표준편차와 분산을 출력하시오
a = np.array([1,2,4,5,5,7,10,13,18,21])
print('표준편차: ', np.std(a), '분산', np.var(a))

# 문제 6. 아래의 배열식을 numpy로 구현하시오
a = np.array([[1,3,7], [1,0,0]])
b = np.array([[0,0,5], [7,5,0]])
print(a + b)

# 문제 7. 아래의 numpy배열을 생성하고 원소중에 '10'만 출력하시오
a = np.array([[1,2,3], [4,10,6], [8,9,20]])
print(a[1, 1])

# 문제 8(점심시간 문제). 아래의 배열연산을 구현하시오
a = np.array([[1,2], [3,4]])
b = np.array([10,20])
print(a * b)

# 문제 9. 아래의 그림의 배열을 numpy로 구현하시오
a = np.array([[0], [10], [20], [30]])
b = np.array([0,1,2])
print(a + b)

# 문제 10. 아래의 배열식을 numpy로 구현하시오
a = np.array([[51,55], [14,19], [0,4]])
print(a[a>=15])

a = a.flatten() # 1차원 배열로 변환
print(a[a>=15])

# 문제 11. 아래의 배열식을 numpy를 이용하지 않고 list변수로 구현하고, 아래의 행렬식에서 행의 갯수가 몇개인지 출력하시오!
a = [[1,3,7],[1,0,0]]
print(len(a))

# 문제 12. 아래의 배열식을 numpy를 이용하지 않고 list변수로 구현하고, 열의 갯수가 몇개인지 출력하시오!
a = [[1,3,7],[1,0,0]]
print(a[0])

# 문제 13. 아래의 배열식의 덧셈 연산을 numpy를 이용하지 않고 수행하시오
a = [[1,3,7], [1,0,0]]
b = [[0,0,5], [7,5,0]]
absum = [[0,0,0], [0,0,0]]

for i in range(len(a)):
    for j in range(len(a[0])):
        absum[i][j] = a[i][j] + b[i][j]
print(absum)

# 문제 14. 아래의 배열식을 numpy이용하지 않고 구현하시오
## numpy 이용
a = np.matrix([[1,2], [3,4]])
b = np.matrix([[5,6], [7,8]])
print(a * b)

## list변수 for문 이용
a1 = [[1,2], [3,4]]
b1 = [[5,6], [7,8]]
c1 = [[0,0], [0,0]]

for i in range(len(a1)):
    for j in range(len(a1[0])):
        for k in range(len(a1[0])):
            c1[i][j] += a1[i][k] * b1[k][j]
print(c1)

# 문제 15. 아래의 행렬 연산을 numpy와 list 2가지 방법으로 구현하시오
## numpy 이용
a = np.matrix([[10,20], [30,40]])
b = np.matrix([[5,6], [7,8]])
print(a - b)

## list변수 for문 이용
a = [[10,20], [30,40]]
b = [[5,6], [7,8]]
c = [[0,0], [0,0]]

for i in range(len(a)):
    for j in range(len(a[0])):
        c[i][j] = a[i][j] - b[i][j]
print(c)

# 문제 16. 브로드캐스트를 사용한 연산을 numpy를 이용하지 않는 방법으로 구현하시오
a = [[1,2], [3,4]]
b = [[10,20]]
c = [[0,0], [0,0]]

for i in range(len(a)):
    for j in range(len(a[0])):
        c[i][j] = a[i][j] * b[0][j]
print(c)

# 예제1:
import matplotlib.pyplot as plt

plt.figure() # 객체를 선언한다.
plt.plot([1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1])
plt.show()

# 예제2: numpy배열을 이용하여 그래프 그리기
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0,12,0.01)
print(t)

plt.figure()
plt.plot(t)
plt.show()

# 문제 17. 위의 그리프에 grid(격자)를 추가하시오
t = np.arange(0,12,0.01)
print(t)

plt.figure()
plt.plot(t)
plt.grid()
plt.show()

# 문제 18. 위 그래프에 x축=size, y축=cost 라고 하시오
t = np.arange(0,12,0.01)
print(t)

plt.figure()
plt.plot(t)
plt.grid()
plt.xlabel("size")
plt.ylabel("cost")
plt.show()

# 문제 19. 위 그래프의 전체 제목을 'size & cost'라고 하시오!
t = np.arange(0,12,0.01)
print(t)

plt.figure()
plt.plot(t)
plt.grid()
plt.title("size & cost")
plt.xlabel("size")
plt.ylabel("cost")
plt.show()

# 문제 20. 아래의 numpy배열로 산포도 그래프를 그리시오
x = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([9,8,7,8,7,3,2,4,3,4])

plt.figure()
plt.scatter(x, y)
plt.show()

# 문제 21. 위의 그래프를 라인그래프로 출력하시오
x = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([9,8,7,8,7,3,2,4,3,4])

plt.figure()
plt.plot(x, y)
plt.show()

# 문제 22. 치킨집 년도별 창업건수를 가지고 라인 그래프를 그리시오
import numpy as np
import matplotlib.pyplot as plt

open = np.loadtxt('d:\data\창업건수.csv', skiprows=1, unpack=True, delimiter=',')

x = open[0]
y = open[4]
plt.figure()
plt.plot(x, y)
plt.title("chicken Open per year")
plt.xlabel('YEAR')
plt.ylabel('CNT')
plt.show()

# 문제 23. 폐업건수도 위의 그래프에 겹치도록 출력하시오
import numpy as np
import matplotlib.pyplot as plt

open = np.loadtxt('d:\data\창업건수.csv', skiprows=1, unpack=True, delimiter=',')
close = np.loadtxt('d:\data\폐업건수.csv', skiprows=1, unpack=True, delimiter=',')

ox = open[0]
oy = open[4]
cx = close[0]
cy = close[4]
plt.figure()
plt.plot(ox, oy)
plt.plot(cx, cy)
plt.title("chicken Open per year")
plt.xlabel('YEAR')
plt.ylabel('CNT')
plt.show()

# 문제 24. 위의 그래프에 legend도 출력하시오
import numpy as np
import matplotlib.pyplot as plt

open = np.loadtxt('d:\data\창업건수.csv', skiprows=1, unpack=True, delimiter=',')
close = np.loadtxt('d:\data\폐업건수.csv', skiprows=1, unpack=True, delimiter=',')

ox = open[0]
oy = open[4]
cx = close[0]
cy = close[4]
plt.figure()
plt.plot(ox, oy, label='open')
plt.plot(cx, cy, label='close')
plt.title("chicken Open per year")
plt.xlabel('YEAR')
plt.ylabel('CNT')
plt.legend()
plt.show()

# 문제 25. 그래프에 이미지를 표시하시오
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('d:\data\lena.png')

plt.imshow(img)
plt.show()

# 문제 26. 강아지 사진으로 출력하시오
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('d:\data\dog.jpg')

plt.imshow(img)
plt.show()
