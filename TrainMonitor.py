# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:33:11 2022

@author: Xinnze
"""
# =============================================================================
# tf.train.MonitoredTrainingSession直接实现保存及载入检查点模型的文件
# 通过指定save_checkpoint_secs参数的具体秒数，来设置每训练多久保存一次检查点
# =============================================================================
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.reset_default_graph()
global_step = tf.train.get_or_create_global_step()
step = tf.assign_add(global_step, 1)
# 设置检查点路径为log/checkpoints
with tf.train.MonitoredTrainingSession(checkpoint_dir="E:\\机器学习_myself\\深度学习\\数据存储\\线性回归\\checkpoints", 
                                       save_checkpoint_secs=2) as sess:
    print(sess.run([global_step]))
    while not sess.should_stop():  # 启用死循环, 当sess不结束时就不停止
        i = sess.run(step)
        print(i)
