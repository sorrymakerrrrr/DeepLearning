# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:28:42 2022
使用函数封装库
@author: Xinnze
"""

import tensorflow.compat.v1 as tf 
from cifar10 import cifar10_input
import numpy as np


tf.disable_v2_behavior()
print('begin')
data_dir = 'cifar-10-batches-bin'

images_train, labels_train = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=128)
images_test, labels_test = cifar10_input.inputs(eval_data=False, data_dir=data_dir, batch_size=128)
print('begin data')


# # 定义权重
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)


# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)


# 定义占位符
x = tf.placeholder(tf.float32, shape=[None, 24, 24, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 24, 24, 3])


# 卷积使用tf.contrib.layers.conv2d  
# 池化使用tf.contrib.layers.max_pool2d tf.contrib.layers.avg_pool2d
# 全连接函数 tf.contrib.layers.fully_connected
# 第一层
h_conv1 = tf.contrib.layers.conv2d(x_image, 64, 5, 1, 'SAME', activation_fn=tf.nn.relu)
h_pool1 = tf.contrib.layers.max_pool2d(h_conv1, [2, 2], stride=2, padding='SAME')
# 第二层
h_conv2 = tf.contrib.layers.conv2d(h_pool1, 64, [5, 5], 1, 'SAME', activation_fn=tf.nn.relu)
h_pool2 = tf.contrib.layers.max_pool2d(h_conv2, [2, 2], stride=2, padding='SAME')
# 全局池化 + 扁平化
nt_hpool2 = tf.contrib.layers.avg_pool2d(h_pool2, [6, 6], stride=6, padding='SAME')
nt_hpool2_flat = tf.reshape(nt_hpool2, [-1, 64])

y_conv = tf.contrib.layers.fully_connected(nt_hpool2_flat, 10, activation_fn=tf.nn.softmax)

cross_entropy = - tf.reduce_sum(y * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuarcy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(session=sess)

for i in range(15000):
    image_batch, label_batch = sess.run([images_train, labels_train])
    label_b = np.eye(10)[label_batch]
    
    train_step.run(feed_dict={x: image_batch, y:label_b})
    
    if i % 200 == 0:
        loss = cross_entropy.eval(feed_dict={x: image_batch, y: label_b})
        train_accuracy = accuarcy.eval(feed_dict={x: image_batch, y: label_b})
        
        print('Epoch: %5d, loss: %g, train_accuracy: %g' % (i, loss, train_accuracy))

print('Finished! ')
