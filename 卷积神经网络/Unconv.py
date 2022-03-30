# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 18:27:42 2022
反卷积操作
结果: 当padding为"SAME"且不需要补0时，卷积和反卷积对于padding为"SAME"和"VALID"都是相同的
@author: Xinnze
"""

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
# 模拟数据
img = tf.Variable(tf.constant(1.0, shape=[1, 4, 4, 1]))

fliter = tf.Variable(tf.constant([1.0, 0, -1, -2], shape=[2, 2, 1, 1]))

# 分别进行valid与same操作
conv = tf.nn.conv2d(img, fliter, strides=[1, 2, 2, 1], padding='VALID')
cons = tf.nn.conv2d(img, fliter, strides=[1, 2, 2, 1], padding='SAME')
print(conv.shape)
print(cons.shape)

# 再进行反卷积
# =============================================================================
# conv2d_transpoes(value, fliter, output_shape, strides, padding="SAME", data_format="NHWC", name=None)
# value: 卷积操作之后的张量 一般用NHWC类型 number height width channel
# fliter: 卷积核
# output_shape: 代表输出的张量形状也是一个四维向量
# strides: 步长
# padding: 代表 原数据 生成value时使用的补0方式  用来检查输入形状与输出形状是否合规
# =============================================================================
contv = tf.nn.conv2d_transpose(conv, fliter, output_shape=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='VALID')
conts = tf.nn.conv2d_transpose(cons, fliter, output_shape=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('fliter: \n', sess.run(fliter))
    print('conv: \n', sess.run(conv))
    print('cons: \n', sess.run(cons))
    print('contv: \n', sess.run(contv))
    print('conts: \n', sess.run(conts))
