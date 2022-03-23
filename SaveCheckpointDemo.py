# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 21:02:33 2022

@author: Xinnze
"""

import tensorflow.compat.v1 as tf
import numpy as np
# import matplotlib.pyplot as plt

# 定义生成loss的可视化函数
tf.disable_v2_behavior()
plotdata = {"batch_size": [], "loss": []}

def moving_average(a, w=10):
    if len(a) < w:
        return a[: ]
    else:
        return [val if idx < w else sum(a[(idx-w): idx]) / w for idx, val in enumerate(a)]
    
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
tf.reset_default_graph()

X = tf.placeholder("float")
Y = tf.placeholder("float")
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")
z = tf.multiply(W, X) + b

cost = tf.reduce_mean(tf.square(Y - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
training_epochs = 20
display_step = 2
saver = tf.train.Saver(max_to_keep=1)  # 生成saver
savedir = savedir = "E:\\机器学习_myself\\深度学习\\数据存储\\线性回归\\"  

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
            
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: x, Y: y})
            print("Epoch: ", epoch+1, " cost: ", loss, 
                  " W: ", sess.run(W, feed_dict={X: x, Y: y}), 
                  " b: ", sess.run(b, feed_dict={X: x, Y: y}))
            if loss != "NA":
                plotdata['batch_size'].append(epoch)
                plotdata['loss'].append(loss)
            # 建立检查点, global_step将训练的次数作为后缀加入到模型名字中
            saver.save(sess, savedir + "linermodel.cpkt", global_step=epoch)
    print("Finished!")

# 重启session 载入检查点
load_epoch = 18
with tf.Session() as sess2:
    saver.restore(sess2, savedir + "linermodel.cpkt-" + str(load_epoch))
    print("x: 15, y_pridict/z: ", sess2.run(z, feed_dict={X: 15}))
