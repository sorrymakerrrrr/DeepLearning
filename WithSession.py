# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 12:50:48 2022

@author: Xinnze
"""

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
a = tf.constant(3)  # 定义常量3
b = tf.constant(4)
with tf.Session() as sess:   # 建立session
    print("相加: %i"  %sess.run(a+b))
    print("相乘: %i"  %sess.run(a*b))


# 使用注入机制，将具体的实参注入到相应的placeholder中。feed只在调用它的方法内有效，方法结束后feed就会消失
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
multiply = tf.multiply(a, b)
with tf.Session() as sess:
    print("相加: %i"  %sess.run(add, feed_dict={a: 3,b: 4}))
    print("相乘: %i"  %sess.run(multiply, feed_dict={a: 3,b: 4}))
    # 一次将多个节点取出
    print(sess.run([add, multiply], feed_dict={a: 3,b: 4}))
    