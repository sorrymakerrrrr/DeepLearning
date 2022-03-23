# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 13:26:39 2022

@author: Xinnze
"""
import tensorflow.compat.v1 as tf


# 指定特定的GPU执行操作
with tf.Session() as sess:
    with tf.device("/gpu:0"):
        x = tf.placeholder(tf.int16)
        y = tf.placeholder(tf.int16)
        add = tf.add(x, y)
        print("相加: %i" %sess.run(add, feed_dict={x: 3, y: 4}))


# =============================================================================
# 通过tf.ConfigProto来构建一个config，在config中指定相关的GPU，并在tf.Session中传入config参数来指定GPU操作
# tf.ConfigProto函数参数如下:
# log_device_placement=True 是否打印设备分配日志
# allow_soft_placement=True 如果指定设备不存在，允许TF自动分配设备
# =============================================================================
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True  # GPU资源按需分配
with tf.Session(config=config) as sess:
    x = tf.placeholder(tf.int16)
    y = tf.placeholder(tf.int16)
    add = tf.add(x, y)
    print("相加: %i" %sess.run(add, feed_dict={x: 3, y: 4}))
