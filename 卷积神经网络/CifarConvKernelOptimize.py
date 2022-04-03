# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 20:21:25 2022
卷积核优化
@author: Xinnze
"""

import tensorflow.compat.v1 as tf
import numpy as np
from cifar10 import cifar10_input
import tensor


tf.disable_v2_behavior()

print('begin')

data_dir = 'cifar-10-batches-bin'
images_train, labels_train = cifar10_input.inputs(eval_data=False, data_dir=data_dir, batch_size=128)
images_test, labels_test = cifar10_input.inputs(True, data_dir, 128)

print('begin data')


def weight_variable(shape):
     initial = tf.truncated_normal(shape, stddev=0.1)
     return tf.Variable(initial)
 

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def avg_pool_6x6(x):
    return tf.nn.avg_pool(x, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 24, 24, 3])
y = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(x, [-1, 24, 24, 3])
W_conv1 = weight_variable([5, 5, 3, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# 演示: 优化卷积核技术。由公式保证5x1的矩阵和1x5的矩阵正好可以生成5x5的矩阵
W_conv21 = weight_variable([5, 1, 64, 64])
b_conv21 = bias_variable([64])
h_conv21 = tf.nn.relu(conv2d(h_pool1, W_conv21) + b_conv21)

W_conv2 = weight_variable([1, 5, 64, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv21, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_conv3 = weight_variable([5, 5, 64, 10])
b_conv3 = bias_variable([10])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

nt_hpool3 = avg_pool_6x6(h_conv3)  # shape 为(-1, 1, 1, 10)
nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
# nt_hpool3_flat = nt_hpool3

y_conv = tf.nn.softmax(nt_hpool3_flat)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y* tf.log(y_conv), 1))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

y_pred = tf.argmax(y_conv, 1)
correct_prediction = tf.equal(y_pred, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess = tf.Session()

sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)  # 启动队列

for i in range(15000):
    image_batch, label_batch = sess.run([images_train, labels_train])
    label_b = np.eye(10)[label_batch]  # 将label_batch转成 独热编码
    
    train_step.run(feed_dict={x: image_batch, y: label_b}, session=sess)
    
    if i % 200 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: image_batch, y: label_b},session=sess)
        loss = cross_entropy.eval(feed_dict={x: image_batch, y: label_b}, session=sess)
        
        print('step %d, loss %g, train_accuary %g' % (i, loss, train_accuracy))
        # print(nt_hpool3.shape, '\n', nt_hpool3_flat.shape)
        # print(x.shape, x_image.shape)

acc_testavg = 0
for i in range(10):
    image_batch, label_batch = sess.run([images_test, labels_test])
    label_b = np.eye(10, dtype=float)[label_batch]
    acc_test = accuracy.eval(feed_dict={x: image_batch, y: label_b}, session=sess)
    acc_testavg += 1 / 10 * acc_test
    
print('Finished!, test accuracy %g' % acc_testavg)

