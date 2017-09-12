# tensorflow 2층 신경망으로 구축하시오!
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("TensorFlow_firststep/MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
W1 = tf.Variable(tf.zeros([784, 50] ))
b1 = tf.Variable(tf.zeros([50]))
z1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.zeros([50, 10]))
b2 = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(z1, W2) + b2)
y_ = tf.placeholder("float", [None, 10])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
# 모든 변수를 초기화하고 세션을 시작한다.
sess.run(tf.global_variables_initializer())

print(sess.run(W1))
print(sess.run(b1))
print(sess.run(W2))
print(sess.run(b2))


for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

# 문제 11. 다중 계층 신경망
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("TensorFlow_firststep/MNIST_data/", one_hot=True)

x = tf.placeholder("float", shape=[None, 784]) # 입력값을 담을 변수
y_ = tf.placeholder("float", shape=[None, 10]) # 라벨을 담을 변수

x_image = tf.reshape(x, [-1,28,28,1]) # 입력이미지를 CNN에 입력하기 위해 reshape
print("x_image=")
print(x_image)

def weight_variable(shape): # 가중치의 값을 초기화하는 함수
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape): # 편향의 값을 초기화하는 함수
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

W_conv1 = weight_variable([5,5,1,32])
# 가로5, 세로5, input 채널1, outpur 채널32로 하는 가중치 매개변수 생성
# feature map을 32개 생성하겠다.
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# x_image 입력값과 가중치의 합성곱에 편향을 더한 값이 relu함수를 통과
h_pool1 = max_pool_2x2(h_conv1) # 14x14
# conv층 통과한 결과가 풀링 계층을 통과

W_conv2 = weight_variable([5,5,32,64])
# 두 번째 conv 계층에 쓰일 가중치 가로5, 세로5, 입력값32
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) # 7x7

W_fc1 = weight_variable([7*7*64, 1024])
# Affine 계층에서 쓰일 가중치 매개변수를 생성
# 가로7, 세로7, feature map 64개, 노드의 개수 1024개
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
# 다시 완전 연결계층에 입력되어야 하므로 텐써를 벡터로 변환한다.
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 벡터로 변환한 결과를 가지고 Afifne을 수행한다.

keep_prob = tf.placeholder("float")
# 소프트맥스 계층전에 dropout 함수를 사용하여 dropout을 적용한다.
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# dropout 1.0: 수행하지 않는다. // 0.5: 수행한다.

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 오차함수에 오차를 최소화하는 경사감소법으로 Adam을 사용하고 있다.
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# 실제 라벨과 예상 라벨을 비교한 결과를 담는다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# [True, False, True, True] --> [1,0,1,1]로 변경하고 차원 축소 후 평균값 출력

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = mnist.train.next_batch(100)
    print(sess.run(W_conv1))
    print(sess.run(b_conv1))
    if i%100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# test_x, test_y = mnist.test.next_batch(1000)
print("test accuracy %g"% sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# 문제 12. 위의 코드를 이해하기 위해 디버깅을 수행하시오
print(h_pool1.get_shape()) # (?, 14, 14, 32)
print(h_pool2.get_shape()) # (?, 7, 7, 64)

# 문제 13. 가중치와 편향의 값 변화를 확인하시오.
for i in range(2):
    batch = mnist.train.next_batch(100)
    print(sess.run(W_conv1))
    print(sess.run(b_conv1))
    if i%100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


