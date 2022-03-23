# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 12:42:42 2022

@author: Xinnze
"""

import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()
hello = tf.constant('Hello,TensorFlow!')  # 定义常量
sess = tf.Session()  # 建立一个session(会话)
print(sess.run(hello))
sess.close()
