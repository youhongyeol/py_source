import copy
import numpy as np

x = np.array([[1.0,-0.5], [-2.0,3.0]])
print(x)

mask = (x <= 0)
print(mask)

out = copy.copy(x) # x와 동일한 주소를 바라보는 참조 객체 out을 생성함
print(x)
print(out)
out[mask] = 0 # 0이하인것을 다 0으로 변경하는 작업

print(out)
print(x)

# 문제 99. 곱셈 계층을 파이썬으로 구현하시오
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

# 문제 100. 위에 구현한 곱셈 계층을 객체화 시켜서 아래의 사과 가격의 총 가격을 구하시오
mul = MulLayer()
apple = 200
apple_cnt = 5
tax = 1.2

apple_price = mul.forward(apple, apple_cnt)
price = mul.forward(apple_price, tax)
print(price)

# 문제 101. 덧셈 계층을 파이썬으로 구현하시오
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

# 문제 102. 사과 2개와 귤 5개를 구입하면 총 가격이 얼마인지 구하시오
apple = 200
apple_cnt = 2
citrus = 300
citrus_cnt = 5
tax = 1.5

# 계층
mul_apple_layer = MulLayer()
mul_citrus_layer = MulLayer()
add_apple_citrus_layer = AddLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_cnt)
citrus_price = mul_citrus_layer.forward(citrus, citrus_cnt)
all_price = add_apple_citrus_layer.forward(apple_price, citrus_price)
price = mul_tax_layer.forward(all_price, tax)
print(price)

# 문제 103. 위의 문제를 역전파를 출력하시오
dprice = 1
dall_prece, dtax = mul_tax_layer.backward(dprice)
dapple_price, dcitrus_price = add_apple_citrus_layer.backward(dall_prece)
dapple, dapple_cnt = mul_apple_layer.backward(dapple_price)
dcitrus, dcitrus_cnt = mul_citrus_layer.backward(dcitrus_price)

print(price)
print(dapple_cnt, dapple, dcitrus_cnt, dcitrus, dtax)

# 문제 104. 사과, 귤, 배 그림 그래프
apple = 100
apple_c = 4
citrus = 200
citrus_c = 3
pear = 300
pear_c = 2
tax = 1.3

# 문제 105. 위의 변수로 순전파, 역전파를 출력하시오
#계층
mul_apple_layer = MulLayer()
mul_citrus_layer = MulLayer()
mul_pear_layer = MulLayer()
apple_citrus_add_layer = AddLayer()
all_add_layer = AddLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_p = mul_apple_layer.forward(apple, apple_c)
citrus_p = mul_citrus_layer.forward(citrus, citrus_c)
pear_p = mul_pear_layer.forward(pear, pear_c)
apple_citrus_p = apple_citrus_add_layer.forward(apple_p, citrus_p)
all_p = all_add_layer.forward(apple_citrus_p, pear_p)
price = mul_tax_layer.forward(all_p, tax)
print(price)

# 역전파
dprice = 1
dall_p, dtax = mul_tax_layer.backward(dprice)
dapple_citrus_p, dpear_p = all_add_layer.backward(dall_p)
dapple_p, dcitrus_p = apple_citrus_add_layer.backward(dapple_citrus_p)
dapple, dapple_c = mul_apple_layer.backward(dapple_p)
dcitrus, dcitrus_c = mul_citrus_layer.backward(dcitrus_p)
dpear, dpear_c = mul_pear_layer.backward(dpear_p)
print(dapple, dapple_c, dcitrus, dcitrus_c, dpear, dpear_c, dtax)

a = [1,2,3,4]
b = a # 단순 복제는 완전히 동일한 객체를 바라보는 것
print(b) # [1,2,3,4]
b[2] = 100
print(b) # [1, 2, 100, 4]
print(a) # [1, 2, 100, 4]

a = 10
b = a
print(b) # 10
b = "abc"
print(b) # abc
print(a) # 10
# 설명: 리스트는 같이 수정이 되지만, 문자열이나 숫자열은 같이 수정이 되지 않고 변수에 값이 할당되어 버린다.

import copy
a = [1,[1,2,3]]
print(a) # [1, [1, 2, 3]]
b = copy.copy(a)
print(b) # [1, [1, 2, 3]]
b[0] = 100
print(b) # [100, [1, 2, 3]]
print(a) # [1, [1, 2, 3]]

b[1][0] = 200
print(b) # [100, [200, 2, 3]]
print(a) # [1, [200, 2, 3]]

a = [1,[2,[3,[4]]]]
b = copy.copy(a)
c = copy.deepcopy(a)
b[1][1][1] = 300
c[1][1][1] = 300
print(b)
print(c)
print(a)

import copy
import numpy as np

x = np.array([[1.0,-0.5], [-2.0,3.0]])
print(x)
# [[ 1.  -0.5]
#  [-2.   3. ]]

mask = ( x <= 0)
print(mask)
# [[False  True]
#  [ True False]]

out = x.copy()
print(out)
# [[ 1.  -0.5]
#  [-2.   3. ]]

out[mask] = 0 # 0이하인것은 모두 0으로 변경해주는 작업
print(out)
# [[ 1.  0.]
#  [ 0.  3.]]
print(x) # copy가 되었기 때문에 별도의 객체인 out이 만들어지고 x객체와는 별도로 out을 변경한 것이다.
# [[ 1.  -0.5]
#  [-2.   3. ]]

# 문제 107. ReLU 함수를 파이썬으로 구현하시오
class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backforard(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

# 문제 108. x 변수를 생성하고 x변수를 ReLU객체의 forward함수에 넣어 출력값을 확인하시오
import numpy as np
x = np.array([[1.0,-0.5], [-2.0,3.0]])
relu = ReLU()
print(relu.forward(x))

# 문제 109. 시그모이드 객체를 구현하시오!
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (0.1 - self.out) * self.out
        return dx