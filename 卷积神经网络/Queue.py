# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 17:18:21 2022

@author: Xinnze
"""
import tensorflow.compat.v1 as tf


tf.disable_v2_behavior()


# 创建长度为100的队列
queue = tf.FIFOQueue(100, "float")

c = tf.Variable(0.0)  # 计数操作
op = tf.assign_add(c, tf.constant(1.0))  # 加1操作

# 操作 将计数器的结果加入队列
enqueue_op = queue.enqueue(c)

# 创建一个队列管理器QueueRunner 用这两个操作向q添加元素。目前只有一个线程 
# QueueRunner类用来启动tensor的入队线程
qr = tf.train.QueueRunner(queue, enqueue_ops=[op, enqueue_op])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    coord = tf.train.Coordinator()
    
    # 启动入队进程，Coordinator是线程的参数
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    
    for i in range(0, 10):
        print("--------------------------------------------------------------------------------")
        print(sess.run(queue.dequeue()))
        
    coord.request_stop()
