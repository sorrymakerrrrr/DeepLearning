# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:18:45 2022

@author: Xinnze
"""

import tensorflow.compat.v1 as tf
import numpy as np
import pylab

tf.disable_v2_behavior()
print("Starting 2nd session...")
saver = tf.train.Saver()
model_path = "E:\\机器学习_myself\\深度学习\\mnist\\521model.ckpt"
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 恢复模型变量
    saver.restore(sess, model_path)
    # 测试model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # 此行可以不要 因为已经保存了
    
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 此行可以不要 因为已经保存了
    print("Accuracy: ", accuracy.eval({x: x_test1, y: y_test_onehot}))
    
    output = tf.argmax(pred, 1)
    index = [np.random.randint(0, len(x_train1)) for _ in range(2)]
    batch_xs = x_train1[index, : ]
    batch_ys = y_train_onehot[index, : ]
    outputval, predv = sess.run([output, pred], feed_dict={x: batch_xs, y: batch_ys})
    print(outputval, predv, tf.argmax(batch_ys, 1).eval())
    
    
    for i in range(2):
        im = batch_xs[i]
        im = im.reshape(-1, 28)
        pylab.imshow(im)
        pylab.show()
