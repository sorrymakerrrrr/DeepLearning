# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 16:18:31 2022
做归一化 BN算法
在tensorflow2.x版本中，tensorflow.contrib.layers.python.layers 中的batch_norm已经移除
所以在tensorflow.campat.v1环境下，编写封装函数实现batch_norm
@author: Xinnze
"""

import tensorflow.compat.v1 as tf 
import numpy as np 
from cifar10 import cifar10_input 


data_dir = 'cifar-10-batches-bin'
images_train, labels_train = cifar10_input.inputs(True, data_dir, 128)
images_test, labels_test = cifar10_input.inputs(False, data_dir, batch_size=128)

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

def batch_norm_layer(inputs, is_training=None, decay=0.9):                
    if is_training == 1:
        return tf.layers.batch_normalization(inputs, momentum=decay, scale=False, training=True)
    else:
        return tf.layers.batch_normalization(inputs, momentum=decay, scale=False, training=False)
    

x = tf.placeholder(tf.float32, [None, 24, 24, 3])
y = tf.placeholder(tf.float32, [None, 10])
train = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, 24, 24, 3])
W_conv1 = weight_variable([5, 5, 3, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 加入BN算法 归一化
W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(batch_norm_layer((conv2d(h_pool1, W_conv2) + b_conv2),train))
h_pool2 = max_pool_2x2(h_conv2)


W_conv3 = weight_variable([5, 5, 64, 10])
b_conv3 = bias_variable([10])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

nt_hpool3 = avg_pool_6x6(h_conv3)  # shape 为(-1, 1, 1, 10)
nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
# nt_hpool3_flat = nt_hpool3

y_conv = tf.nn.softmax(nt_hpool3_flat)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y* tf.log(y_conv), 1))

global_step = tf.Variable(0, trainable=False)
decay_learning_rate = tf.train.exponential_decay(0.004, global_step, 1000, 0.9)

# 损失函数优化器的minimize()中global_step=global_steps能够提供global_step自动加一的操作。
# global_steps是等号右边，在编程语言里面通常当作定值（即不会受函数影响）赋值给等号左边的global_step。
# 然而，在这个优化器里面能够使得右边的变量自动加一。这确实是编程语言里面少见的，也是需要特别注意的。
train_step = tf.train.AdamOptimizer(decay_learning_rate).minimize(cross_entropy, global_step=global_step)

y_pred = tf.argmax(y_conv, 1)
correct_prediction = tf.equal(y_pred, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()

sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)  # 启动队列

for i in range(6000):
    image_batch, label_batch = sess.run([images_train, labels_train])
    label_b = np.eye(10)[label_batch]  # 将label_batch转成 独热编码
    
    train_step.run(feed_dict={x: image_batch, y: label_b, train: 1}, session=sess)
    
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
