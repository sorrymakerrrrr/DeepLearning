# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:12:00 2022
sobel
@author: Xinnze
"""


import tensorflow.compat.v1 as tf 
import numpy as np
import matplotlib.pyplot as plt  # 用于显示图片
import matplotlib.image as mpimg  # 用于读取图片

# 1.载入图片并显示
myimg =mpimg.imread('img.jpg')
# plt.imshow(myimg)
# plt.axis('off')
# plt.show()
# print(myimg.shape)

# 2.定义占位符、卷积核、卷积op
# 手动将sobel算子填入到卷积核中。
full = np.reshape(myimg, [1, 4607, 3455, 3])
# 使用tf.constant函数可以将常量直接初始化到Variable中
inputfull = tf.Variable(tf.constant(1.0, shape=[1, 4607, 3455, 3]))

# 因为是三通道，所以sobel卷积核每个元素都被扩成了3个
fliter = tf.Variable(tf.constant([[-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0],
                                  [-2.0, -2.0, -2.0], [0, 0, 0], [2.0, 2.0, 2.0],
                                  [-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0],],
                                 shape=[3, 3, 3, 1]))

# 每一个in_channel的卷积核均为[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

# 三个通道输入，生成一个feature map
op = tf.nn.conv2d(inputfull, fliter, strides=[1, 1, 1, 1], padding='SAME')

# sobel算子处理过后的图片不保证每个像素都在0-255之间，所以要做一次归一化操作，让生成的值都在[0, 1]之间，然后再乘以255
# 归一化操作：用每一个值减去最小值的结果，再除以最大值与最小值的差  最大值为[1, 4607, 3455, 1]所有元素的最大值
o = tf.cast(((op - tf.reduce_min(op)) / (tf.reduce_max(op) - tf.reduce_min(op))) * 255, tf.uint8)

# 3.运行卷积操作并且显示
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t, f = sess.run([o, fliter], feed_dict={inputfull: full})
    
    t = np.reshape(t, [4607, 3455])
    plt.imshow(t, cmap='Greys_r')
    plt.axis('off')
    # 一张提取到轮廓特征的图像
    plt.show()
