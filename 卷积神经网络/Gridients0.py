# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:47:06 2022
understand gradient
@author: Xinnze
"""

import tensorflow.compat.v1 as tf


tf.disable_v2_behavior()


# gradients不支持 int型的Tensor的gradients 
# 把w1的设置成float类型的例如tf.float32 gards就能算了
# 而且tensorflow梯度值一般都是float32类型的 ???
w1 = tf.Variable([[1, 2]], dtype=tf.float32)
w2 = tf.Variable([[3, 4]], dtype=tf.float64)

y = tf.matmul(w1, [[9], [10]])
# 如果梯度的式子中没有要求的偏导的变量，系统会报错
grads = tf.gradients(y, [w1])  # 求w1的梯度

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gradval = sess.run(grads)
    print(gradval)
