# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:49:09 2022
pool function
@author: Xinnze
"""

import tensorflow.compat.v1 as tf

# =============================================================================
# tf.nn.nax_pool(input, ksize, strides, padding, name=None)
# (avg_pool)
# input: 输入通常为feature map 即 [batch, height, width, channels]
# ksize: 池化窗口的大小，取一个四维向量，一般为[1, height, width, 1]
# strides: 同卷积参数
# padding: 同卷积参数
# 返回一个Tensor，类型不变 依然是[batch, height, width, channels]
# =============================================================================

# 1.定义输入变量
img = tf.constant([
    [[0.0, 4.0], [0.0, 4.0], [0.0, 4.0], [0.0, 4.0]],
    [[1.0, 5.0], [1.0, 5.0], [1.0, 5.0], [1.0, 5.0]],
    [[2.0, 6.0], [2.0, 6.0], [2.0, 6.0], [2.0, 6.0]],
    [[3.0, 7.0], [3.0, 7.0], [3.0, 7.0], [3.0, 7.0]]
    ])
img = tf.reshape(img, [1, 4, 4, 2])

# 2.定义池化操作
pooling = tf.nn.max_pool(img, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
pooling1 = tf.nn.max_pool(img, [1, 2, 2, 1], [1, 1, 1, 1], padding='VALID')
pooling2 = tf.nn.avg_pool(img, [1, 4, 4, 1], [1, 1, 1, 1], padding='SAME')

# 全局池化法，使用一个与原有输入同样尺寸的fliter进行池化，一般放在最后一层
# 用于表达图像通过卷积网络处理后的最终特征
pooling3 = tf.nn.avg_pool(img, [1, 4, 4, 1], [1, 4, 4, 1], padding='SAME')

# 将数据转置后的均值操作得到的值与全局池化平均值是一样的结果
nt_hpool2_flat = tf.reshape(tf.transpose(img), [-1, 16])
pooling4 = tf.reduce_mean(nt_hpool2_flat, 1)

with tf.Session() as sess:
    print('image:')
    image = sess.run(img)
    print(image)
    
    result = sess.run(pooling)
    print("result:\n", result)
    
    result1 = sess.run(pooling1)
    print("result1: \n", result1)
    
    result2 = sess.run(pooling2)
    print("result2: \n", result2)
    
    result3 = sess.run(pooling3)
    print("result3: \n", result3)
    
    flat, result4 = sess.run([nt_hpool2_flat, pooling4])
    print("result4: \n", result4)
    print("flat: \n", flat)
    