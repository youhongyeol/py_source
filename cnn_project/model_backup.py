import tensorflow as tf
import numpy as np

# 정규화, label one-hot encoding
def data_setting(data):
    # x : 데이터, y : 라벨
    x = (np.array(data[:, 0:-1]) / 255).tolist()
    y_tmp = np.zeros([len(data), 5])
    for i in range(0, len(data)):
        label = int(data[i][-1])
        y_tmp[i, label - 1] = 1
    y = y_tmp.tolist()

    return x, y

# 배치 정규화
def BN(input, training, name, scale=True, decay=0.99):
    return tf.contrib.layers.batch_norm(input, decay=decay, scale=scale,
                                        is_training=training, updates_collections=None, scope=name)

print('데이터 로드 중..')

train_data = np.loadtxt('d:/python/data/train5.csv', delimiter=',')
test_data = np.loadtxt('d:/python/data/test5.csv', delimiter=',')

x_train, t_train = data_setting(train_data)
x_test, t_test = data_setting(test_data)

# numpy 로 변환
x_train, t_train = np.array(x_train), np.array(t_train)
x_test, t_test = np.array(x_test), np.array(t_test)

# 원하는 수의 데이터만 배치
x_train, t_train = x_train[:100], t_train[:100]
x_test, t_test = x_test[:100], t_test[:100]

print('데이터 로드 완료!')

print(len(x_train))

# 하이퍼 파라미터!
learning_rate = 0.001
training_epochs = 20
batch_size = 10

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)
            self.X = tf.placeholder(tf.float32, [None, 1024])
            X_img = tf.reshape(self.X, [-1, 32, 32, 1])
            self.Y = tf.placeholder(tf.float32, [None, 5])

            # Convolutional Layer #1 & Pooling Layer #1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     padding="SAME", activation=None)
            conv1_bn = BN(input=conv1, training=1, name='conv1_bn')
            conv1_bn_rl = tf.nn.relu(conv1_bn, name='conv1_bn_rl')
            pool1 = tf.layers.max_pooling2d(inputs=conv1_bn_rl, pool_size=[2, 2], padding="SAME", strides=2)
            print('pool1.shape', pool1.shape)

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     padding="SAME", activation=tf.nn.relu)
            conv2_bn = BN(input=conv2, training=1, name='conv2_bn')
            conv2_bn_rl = tf.nn.relu(conv2_bn, name='conv2_bn_rl')
            pool2 = tf.layers.max_pooling2d(inputs=conv2_bn_rl, pool_size=[2, 2], padding="SAME", strides=2)
            print('pool2.shape', pool2.shape)

            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     padding="SAME", activation=tf.nn.relu)
            conv3_bn = BN(input=conv3, training=1, name='conv3_bn')
            conv3_bn_rl = tf.nn.relu(conv3_bn, name='conv3_bn_rl')
            pool3 = tf.layers.max_pooling2d(inputs=conv3_bn_rl, pool_size=[2, 2], padding="SAME", strides=2)
            print('pool3.shape', pool3.shape)

            # Convolutional Layer #4 and Pooling Layer #4
            conv4 = tf.layers.conv2d(inputs=pool3, filters=256, kernel_size=[3, 3],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     padding="SAME", activation=tf.nn.relu)
            conv4_bn = BN(input=conv4, training=1, name='conv4_bn')
            conv4_bn_rl = tf.nn.relu(conv4_bn, name='conv4_bn_rl')
            pool4 = tf.layers.max_pooling2d(inputs=conv4_bn_rl, pool_size=[2, 2], padding="SAME", strides=2)
            print('pool4.shape', pool4.shape)

            # Dense Layer with Relu
            flat = tf.reshape(pool4, [-1, 2 * 2 * 256])
            dense4 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
            dropout5 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)

            # Logits (no activation) Layer: -> 5 outputs
            self.logits = tf.layers.dense(inputs=dropout5, units=5)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test,
                                                       self.Y: y_test, self.training: training})

    def etrain(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer],
                             feed_dict={self.X: x_data, self.Y: y_data, self.training: training})

sess = tf.Session()
models = []
num_models = 5
for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))

sess.run(tf.global_variables_initializer())

print('Learning Start')
# train model
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(x_train.shape[0] / batch_size)

    for step in range(0, x_train.shape[0], batch_size):
        batch_xs, batch_ys = np.array(x_train[step:step + batch_size]), np.array(t_train[step:step + batch_size])

        for m_idx, m in enumerate(models):
            c, _ = m.etrain(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch

    print('Epoch: ', '%04d' % (epoch + 1), 'cost = ', avg_cost_list)
print('Learning end')

# Test model and check accuracy
predictions = np.zeros(len(x_test) * 5).reshape(len(x_test), 5)

for m_idx, m in enumerate(models):
    print(m_idx + 1, 'model accuracy: ', m.get_accuracy(x_test, t_test))
    p = m.predict(x_test)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(t_test, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('cifar5 accuracy: ', sess.run(ensemble_accuracy))