# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:19:43 2022
用CNN处理mnist数据集 退化学习率 + BN算法归一化 + 多通道卷积处理 正确率97%
@author: Xinnze
"""

import tensorflow.compat.v1 as tf 
from tensorflow.core.example.tutorials.mnist import input_data



tf.disable_v2_behavior()
tf.reset_default_graph()
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
print(mnist.train.next_batch(128)[0].shape)


def weight_variable(shape):  # [n, n, 1, 32]
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def avg_pool_7x7(x):
    return tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME')


def batch_norm_layer(inputs, is_training=None, decay=0.9):
    if is_training == 1:
        return tf.layers.batch_normalization(inputs, momentum=decay, scale=False, training=True)
    else:
        return tf.layers.batch_normalization(inputs, momentum=decay, scale=False, training=False)
    
    

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
train = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2_1x1 = weight_variable([1, 1, 32, 32])
b_conv2_1x1 = bias_variable([32])
W_conv2_3x3 = weight_variable([3, 3, 32, 32])
b_conv2_3x3 = bias_variable([32])
W_conv2_5x5 = weight_variable([5, 5, 32, 32])
b_conv2_5x5 = bias_variable([32])
W_conv2_7x7 = weight_variable([7, 7, 32, 32])
b_conv2_7x7 = bias_variable([32])


# 多通道卷积 + BN算法归一化
h_conv2_1x1 = tf.nn.relu(batch_norm_layer(conv2d(h_pool1, W_conv2_1x1) + b_conv2_1x1, train))
h_conv2_3x3 = tf.nn.relu(batch_norm_layer(conv2d(h_pool1, W_conv2_3x3) + b_conv2_3x3, train))
h_conv2_5x5 = tf.nn.relu(batch_norm_layer(conv2d(h_pool1, W_conv2_5x5) + b_conv2_5x5, train))
h_conv2_7x7 = tf.nn.relu(batch_norm_layer(conv2d(h_pool1, W_conv2_7x7) + b_conv2_7x7, train))

h_conv2 = tf.concat([h_conv2_1x1, h_conv2_3x3 ,h_conv2_5x5, h_conv2_7x7], 3)
h_pool2 = max_pool_2x2(h_conv2)


W_conv3 = weight_variable([5, 5, 128, 10])
b_conv3 = bias_variable([10])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
nt_hpool3 = avg_pool_7x7(h_conv3)
nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
y_conv = tf.nn.softmax(nt_hpool3_flat)

cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))

global_steps = tf.Variable(0, trainable=False)
# 退化学习率
decay_learning_rate = tf.train.exponential_decay(0.0003, global_steps, 1000, 0.9)
train_step = tf.train.AdamOptimizer(decay_learning_rate).minimize(cross_entropy, global_step=global_steps)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 128
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(5000):        
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        _ = sess.run(train_step, feed_dict={x: x_batch, y: y_batch, train: 1})
        
        if epoch % 200 == 0:
            loss = cross_entropy.eval(feed_dict={x: x_batch, y: y_batch, train: 1})
            print("Epoch: %d, loss: %g" % (epoch, loss))

    
    print("accuracy: %g" % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))
       