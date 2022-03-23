# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:35:24 2022

@author: Xinnze
"""
import tensorflow.compat.v1 as tf
import tensorflow
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w): idx])/w for idx, val in enumerate(a)]


tf.disable_v2_behavior()
# 一.准备数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape)*0.3  # 加入了噪声

# 二.搭建模型
# 2.1.正向搭建模型
# 占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")
# =============================================================================
# 通过字典类型定义输入节点
# inputdict = {
#     'x': tf.placeholder("float")
#     'y': tf.placeholder("float")
#     }
# =============================================================================
# 模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")
# 前向结构
z = tf.multiply(X, W) + b

tf.summary.histogram('z', z)  # 将预测值以直方图形式显示 直方图名字为z

# =============================================================================
# # 字典的方式定义学习参数
# paradict = {
#     'w': tf.Variable(tf.random_normal(1)),
#     'b': tf.Variable(tf.zeros([1]))
#     }
# z = tf.multiply(X, paradict['w']) + paradict['b']
# =============================================================================

# 2.2.反向搭建模型
cost = tf.reduce_mean(tf.square(Y - z))

tf.summary.scalar("loss_function", cost)  # 将损失以标量形式显示 标量名字为loss function

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # 梯度下降算法
# 三.迭代训练模型
# 3.1.训练模型
# 初始化所有变量
init = tf.global_variables_initializer()
# 定义参数
training_epochs = 20
display_step = 2
# 启动session
with tf.Session() as sess:
    sess.run(init)
    
    merged_summary_op = tf.summary.merge_all()  # 合并所有summary
    # 创建summary_writer,用于写文件
    summary_writer = tf.summary.FileWriter("E:\\机器学习_myself\\深度学习\\数据存储\\线性回归\\mnist_with_summaries", sess.graph)
    
    plotdata = {"batchsize": [], "loss": []}
    # 向模型输入数据
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        
        summary_str = sess.run(merged_summary_op, feed_dict={X: x, Y: y})
        summary_writer.add_summary(summary_str, epoch)  # 将summary写入文件 
        
        # 显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", epoch + 1, "cost=", loss, "W=", sess.run(W),
                  "b=", sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
    print(" Finished!")
    print("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))
    # 3.2.训练模型可视化
    plt.figure(1)
    plt.subplot(211)
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fittedline')
    plt.legend()
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.subplot(212)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], color='black', linestyle='--')
    plt.xlabel('Minibatch number')
    plt.ylabel('loss')
    plt.title('Minibatch run vs. Training loss')

    plt.show()

    # 四.使用模型
    print("x=0.2, z=", sess.run(z, feed_dict={X: 0.2}))
