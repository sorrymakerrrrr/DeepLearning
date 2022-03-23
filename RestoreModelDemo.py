# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:46:41 2022

@author: Xinnze
"""

import tensorflow.compat.v1 as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

savedir = "E:\\机器学习_myself\\深度学习\\数据存储\\线性回归\\linemodel.ckpt"
saver = tf.train.Saver()
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())  # 初始化
    saver.restore(sess2, savedir)
    print("x: 0.2, z: ", sess2.run(z, feed_dict={X: 0.2}))

print_tensors_in_checkpoint_file(savedir, None, True)
