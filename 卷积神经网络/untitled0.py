# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 15:09:31 2022

@author: Xinnze
"""

import tensorflow.compat.v1 as tf 

import copy

tf.disable_v2_behavior()

img = tf.constant([
    [[0.0, 4.0], [0.0, 4.0], [0.0, 4.0], [0.0, 4.0]],
    [[1.0, 5.0], [1.0, 5.0], [1.0, 5.0], [1.0, 5.0]],
    [[2.0, 6.0], [2.0, 6.0], [2.0, 6.0], [2.0, 6.0]],
    [[3.0, 7.0], [3.0, 7.0], [3.0, 7.0], [3.0, 7.0]],
    ])
img = tf.reshape(img, [1, 4, 4, 2])

# x = tf.nn.moments(img, axes=[0, 1, 2])

# with tf.Session() as sess:
#    mean, std = sess.run(x)
#    print(mean, std)

w1 = tf.Variable(1.0)
w2 = tf.Variable(2.0)
w3 = tf.Variable(2.0)
w4 = tf.Variable(2.0)
w = [w1, w2, w3, w4]
ema = tf.train.ExponentialMovingAverage(0.9)
pop_mean = tf.Variable(tf.zeros([2]), trainable=False, name='pop-mean')
batch_mean, batch_var = tf.nn.moments(img, [0, 1, 2], name='moments')
avg = pop_mean * 0.9 + batch_mean * 0.1
print(avg)
# update=tf.assign_add(w1,1.0)
# ema_op = ema.apply([w1,w2,w3,w4])

# init=tf.initialize_all_variables()
# with tf.control_dependencies([ema_op, update]):
#     for i in range(3):
#         a = tf.identity([ema.average(j) for j in [w1, w2, w3, w4]])
#         pop_mean = [ema.average(j) for j in [w1, w2, w3, w4]]
#         print(a)
#         print(pop_mean)

