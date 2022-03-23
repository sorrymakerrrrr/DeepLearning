# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:29:42 2022

@author: Xinnze
"""

from tensorflow.core.example.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
# import pylab

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
tf.disable_v2_behavior()

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 定义学习参数
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 定义输出节点
pred = tf.nn.softmax(tf.matmul(x, W) + b)  # softmax分类

# 定义反向传播结构  
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))  # 交叉熵

learning_rate = 0.01 # 定义参数
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)  # 使用梯度下绛优化器

# 训练模型并输出中间状态参数
training_epochs = 25
batch_size = 100  # 在训练过程中一次取100条进行训练
display_step = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化
    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)  # 每一轮迭代需要循环的所有训练次数
        
        # 循环所有数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
# =============================================================================
#             print(batch_xs.shape)
#             print(batch_ys.shape)
# =============================================================================

            
            # 运行优化器
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_xs, y: batch_ys})
            # 计算平均loss值  c为batch_size个训练样本的平均loss值  平均的平均值
            avg_cost += c / total_batch

        # 显示训练的详细信息
        if (epoch + 1) % display_step == 0:
            print("Epoch: ", '%04d' % (epoch + 1), "cost = ", "{:.9f}".format(avg_cost))
        
    print("Finished!")
    
    # 5.5测试模型
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))  # axis=1 取每一行的最大索引
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
