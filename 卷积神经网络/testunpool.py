# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 20:04:06 2022

@author: Xinnze
"""

import tensorflow.compat.v1 as tf 
from Unpooling import max_pool_with_argmax

tf.disable_v2_behavior()

img = tf.constant([
    [[0.0, 4.0], [0.0, 4.0], [0.0, 4.0], [0.0, 4.0]],
    [[1.0, 5.0], [1.0, 5.0], [1.0, 5.0], [1.0, 5.0]],
    [[2.0, 6.0], [2.0, 6.0], [2.0, 6.0], [2.0, 6.0]],
    [[3.0, 7.0], [3.0, 7.0], [3.0, 7.0], [3.0, 7.0]],
    ])
img = tf.reshape(img, [1, 4, 4, 2])

pooling2 = tf.nn.max_pool(img, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
encode, mask = max_pool_with_argmax(img, 2)

with tf.Session() as sess:
    
    print("image: ")
    image = sess.run(img)
    print(image)
    
    result = sess.run(pooling2)
    print("pooling2: \n", result)
    
    result2, mask2 = sess.run([encode, mask])
    

batch_range = tf.reshape(tf.range(1, dtype=tf.int64), shape=[1, 1, 1, 1])
feature_range = tf.range(1, dtype=tf.int64)
one_like_mask = tf.ones_like(mask2)
y = mask2 // (8)
x = mask2 % (8) // 2
# print(25 % 8 // 2)
# print(feature_range)
# print(one_like_mask)
b = one_like_mask * batch_range
f = one_like_mask * feature_range

with tf.Session() as sess:
    # print(sess.run(f))
    print(sess.run((tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, -1])))))
