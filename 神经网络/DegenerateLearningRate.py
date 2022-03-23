# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:39:43 2022

@author: Xinnze
"""
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 0.1  # 初始学习率
learning_rate = tf.train.exponential_decay(initial_learning_rate, 
                                           global_step=global_step,
                                           decay_steps=10,
                                           decay_rate=0.9)
opt = tf.train.GradientDescentOptimizer(learning_rate)
add_global = global_step.assign_add(1)  # 定义一个op,令global_step加1完成计步
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(learning_rate))
    for i in range(20):
        # 循环20步, 将每一步的学习率打印出来
        g, rate = sess.run([add_global, learning_rate])
        print(g, rate)