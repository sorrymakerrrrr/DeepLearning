# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:02:17 2022
understand gradient
使用gradients对多个式子求多变量偏导
@author: Xinnze
"""


import tensorflow.compat.v1 as tf 


tf.disable_v2_behavior()

tf.reset_default_graph()

w1 = tf.get_variable('w1', shape=[2])
w2 = tf.get_variable('w2', shape=[2])

w3 = tf.get_variable('w3', shape=[2])
w4 = tf.get_variable('w4', shape=[2])

y1 = w1 + w2 + w3
y2 = w3 + w4

a = w1 + w2 
a_stoped = tf.stop_gradient(a)  # 令a的梯度停止
y3 = a_stoped + w3 

# =============================================================================
# tf.gradients(ys, xs, 
# 			 grad_ys=None, 
# 			 name='gradients',
# 			 colocate_gradients_with_ops=False,
# 			 gate_gradients=False,
# 			 aggregation_method=None,
# 			 stop_gradients=None)
# =============================================================================

# grad_ys求梯度输入值
# grad_ys长度等于len(ys)。这个参数的意义在于对xs中的每个元素的求导加权重。
gradients = tf.gradients([y1, y2], [w1, w2, w3, w4], grad_ys=[tf.convert_to_tensor([1., 2.]), 
                                                              tf.convert_to_tensor([3., 4.])])

# =============================================================================
# w1 w2对应位置的值均为None 这是由于梯度被停止了。后面的程序试图去求一个None的梯度 所以报错
# gradients2 = tf.gradients(y3, [w1, w2, w3], grad_ys=[tf.convert_to_tensor([1., 2.]), 
#                                                      tf.convert_to_tensor([3., 4.])])
# =============================================================================

gradients3 = tf.gradients(y3, [w3], grad_ys=tf.convert_to_tensor([1, 2]))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(gradients), '\n')
    # print(sess.run(gradients2))
    print(sess.run(gradients3))

