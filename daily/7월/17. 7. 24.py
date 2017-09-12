# 문제 74. 아래의 함수를 미분해서 x = 4에서의 미분계수를 구하시오
# y = x^2 + 4^2
def function_2(x):
    return x**2 + 4**2

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

print(numerical_diff(function_2, 4))
# 7.999999999999119 >> 진정한 미분이 아니라 수치미분이기 때문에 중앛차분 오차가 발생하고 있다.

# 문제 75. 아래의 함수를 시각화 하시오]
import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_2(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

# 문제 76. f(x0,x1) = x0^2 + x1^2 함수를 편미분하는데 x0=3, x1=4일 때
# 수치 미분 함수
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function_tmp1(x0):
    return x0**2 + 4**2

print(numerical_diff(function_tmp1,3)) # 6.00000000000378

# 문제 77. x0=3, x1=4일 때 아래의 함수를 아래와 같이 편미분 하시오
def function_tmp2(x1):
    return 3**2 + x1**2

print(numerical_diff(function_tmp2, 4)) # 7.999999999999119

# 문제 78. 아래의 함수를 x0로 편미분 하시오(X0=3, X1=4)
# f(x0, x1) = 2*x0^2 + 3*x1^2
def function_tmp2(x0):
    return 2*x0**2 + 3*4**2
print(numerical_diff(function_tmp2,3))

func = lambda x0:2*x0**2 + 3*4**2
print(numerical_diff(func, 3))

# 문제 79. 아래의 함수를 x1으로 편미분 하시오 lambda를 이용하세요. (x0=6, x1=7)
# f(x0, x1) = 6*x0^2 + 2*x1^2
func1 = lambda x1: 6*6**2 + 2*x1**2
print(numerical_diff(func1, 7))

# 문제 80(점심시간 문제). for loop 문을 이용해서 위의 함수를 x0로 편미분하고 x1로 편미분을 각각 수행하시오
func0 = lambda x0: 6*x0**2
func1 = lambda x1: 2*x1**2

f = func0, func1
n = [6,7]

for i,j in zip(f,n):
    print(numerical_diff(i, j))

# 문제 81. 위의 편미분을 벡터로 나타내는것을 파이썬으로 구현하시오
import numpy as np
def numericalGradient(f, x): # [3.0, 4.0]
    h = 1e-4
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성
    for idx in range(x.size): # x.size는 2 / 0,1로 loop 수행
        tmpVal = x[idx] # 3.0
        x[idx] = tmpVal + h # [3.0001, 4.0]
        print(idx, x)
        fxh1 = f(x) # 25.0006

        x[idx] = tmpVal - h # [2.9999, 4.0]
        print(idx, x)
        fxh2 = f(x) # 24.9994

        grad[idx] = (fxh1 - fxh2) / (2*h) # (25.0006 - 24.9994) / (2*0.0001)
        x[idx] = tmpVal # 원래 값 복원
    return grad

def f(x):
    return x[0]**2 + x[1]**2

print(numericalGradient(f, np.array([3.0, 4.0])))

# 문제 82. numpy의 zeros_like가 무엇인지 확인해보시오
import numpy as np
x = np.array([3.0, 4.0])
grad = np.zeros_like(x)
print(grad.shape)

# 문제 83. x0=0.0, x1=2.0일 때의 기울기 벡터를 구하시오
print(numericalGradient(f, np.array([0.0, 2.0]))) # 0, 4

# 문제 84. x0=3.0, x1=1.0일 때의 기울기 벡터를 구하시오
print(numericalGradient(f, np.array([3.0, 1.0]))) # 6, 2

print(numericalGradient(f, np.array([1.0, 0.0]))) # 2, 0
# 가장 낮은곳으로 갈수록 화살표의 크기가 작아짐을 확인할 수 있다.

# 문제 85. 경사 감소 함수를 파이썬으로 구현하시오.(비용함수를 미분해서 기울기 0에 가까워지는 그 지검의 기울기를 구하시오)
init_x = np.array([-3.0, 4.0])

def function_2(x):
    return x[0]**2 + x[1]**2

def gradientDescent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numericalGradient(f, x)
        x -= lr * grad
    return x

print(gradientDescent(function_2, init_x=init_x, lr=0.1, step_num=100))
# [ -6.11110793e-10   8.14814391e-10]
#거의 (0,0)에 가까운 결과가 출력이 되었다.

# 문제 86. 위의 식을 그대로 사용해서 테스트를 수행하는데 학습률이 너무 크면(10) 발산하고 학습률이 너무 작으면(1e-10)으로 수렴을 못한다는 것을 확인하시오
print(gradientDescent(function_2, init_x=init_x, lr=10, step_num=100))

print(gradientDescent(function_2, init_x=init_x, lr=1e-10, step_num=100))

# 문제 87(마지막 문제). 러닝 레이트를 1e-10으로 했을 때 기울기가 0으로 수렴하려면 step_num을 몇을 줘야하는지 확인하시오





