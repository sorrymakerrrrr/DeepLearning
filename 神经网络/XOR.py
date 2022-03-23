# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 09:20:06 2022
XOR Operation
@author: Xinnze
"""

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()
# 1.定义变量
learning_rate = 0.01
n_input = 2  # 输入层节点个数
n_label = 1
n_hidden = 2  # 隐藏层节点个数

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_label])

# 2.定义学习参数

# 以字典的方式定义权重w和b 里面的h1代表隐藏层，h2代表最终的输出层

weights = {
    # tf.truncated_normal(shape, mean, stddev)
    # 截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1)),
    'h2': tf.Variable(tf.truncated_normal([n_hidden, n_label], stddev=0.1))
    }
biases = {
    'h1': tf.Variable(tf.zeros([n_hidden])),
    'h2': tf.Variable(tf.zeros([n_label]))
    }

# 3.定义网络模型
layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['h1']))
y_pred = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['h2']), biases['h2']))

# 模型的反向使用均值平方差计算loss，用AdamOptimizer进行优化
loss = tf.reduce_mean((y_pred - y) ** 2)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 4.构建模拟数据
# 生成数据
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
X = np.array(X).astype('float32')
Y = np.array(Y).astype('int16')

# 5.运行session，生成结果
# =============================================================================
# tf.InteractiveSession():它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。可以先构建一个session然后再定义操作
# tf.Session():需要在启动session之前构建整个计算图，然后启动该计算图。需要在会话构建之前定义好全部的操作
# =============================================================================
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# 训练
for i in range(2000):
    sess.run(train_step, feed_dict={x: X, y: Y})
# 计算预测值
print(sess.run(y_pred, feed_dict={x: X}))

# 查看隐藏层的输出
print(sess.run(layer_1, feed_dict={x: X}))
