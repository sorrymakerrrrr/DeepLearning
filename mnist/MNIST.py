# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 09:52:34 2022

@author: Xinnze
"""

import tensorflow as tf2
import pylab
import tensorflow.compat.v1 as tf
import numpy as np


# 5.1 导入MNIST数据集
(x_train, y_train), (x_test, y_test) = tf2.keras.datasets.mnist.load_data()

# =============================================================================
# print(x_train.shape)  # 每张图片像素为28*28
# im = x_train[0]
# pylab.imshow(im)
# pylab.show()
# =============================================================================
tf.disable_v2_behavior()

x_train1 = 1/255 * x_train.reshape(-1, 784).astype(np.float32)  # 将其28*28拉长为一个向量 并且转化为float32类型  ps:归一化很重要！！！
x_test1 = 1/255 * x_test.reshape(-1, 784).astype(np.float32)


y_train_onehot = tf.cast(tf.one_hot(y_train, 10), tf.float32)  # 将y_train变为独热编码
y_test_onehot = tf.cast(tf.one_hot(y_test, 10), tf.float32)
# print(y_train_onehot.shape)

# 5.2分析图片的特点与变量定义
tf.Graph().as_default()

# 定义占位符
x = tf.placeholder(tf.float32, [100, 784])
y = tf.placeholder(tf.float32, [100, 10])

# 定义学习参数
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 定义输出节点
pred = tf.nn.softmax(tf.matmul(x, W) + b)  # softmax分类

# 定义反向传播结构  
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))  # 交叉熵
cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=(tf.matmul(x, W) + b)))

learning_rate = 0.01 # 定义参数
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)  # 使用梯度下绛优化器

# 训练模型并输出中间状态参数
training_epochs = 25
batch_size = 100  # 在训练过程中一次取100条进行训练
display_step = 1

saver = tf.train.Saver()
model_path = "E:\\机器学习_myself\\深度学习\\mnist\\521model.ckpt"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化
    y_train_onehot = y_train_onehot.eval(session=sess)
    
    for epoch in range(training_epochs):
        avg_cost1 = 0
        avg_cost2 = 0
        total_batch = int(len(x_train) / batch_size)  # 每一轮迭代需要循环的所有训练次数
        
        # 循环所有数据集
        for i in range(total_batch):
            # index = [np.random.randint(0, len(x_train)-1) for _ in range(100)]
            # batch_xs = x_train1[index, :]
            # batch_ys = y_train_onehot[index, :]
            batch_xs = x_train1[batch_size*i: batch_size*i + 100, :]
            batch_ys = y_train_onehot[batch_size*i: batch_size*i + 100, :]
# =============================================================================
#             print(batch_xs.shape)
#             print(batch_ys.shape)
# =============================================================================
            
            # 运行优化器
            _, c1, c2 = sess.run([optimizer, cross_entropy, cross_entropy2], feed_dict={x: batch_xs, y: batch_ys})
            # 计算平均loss值  c为batch_size个训练样本的平均loss值  平均的平均值
            avg_cost1 += c1 / total_batch
            avg_cost2 += c2 / total_batch

        # 显示训练的详细信息
        if (epoch + 1) % display_step == 0:
            print("Epoch: ", '%04d' % (epoch + 1), "cost1 = ", "{:.9f}".format(avg_cost1), "cost2 = ", "{:.9f}".format(avg_cost2))
        
    print("Finished!")

    # 5.5测试模型

    # 测试model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # axis=1 取每一行的最大索引
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    y_test_onehot = y_test_onehot.eval(session=sess)
    print("Accuracy: ", accuracy.eval({x: x_test1, y: y_test_onehot}))  # accuracy.eval() = sess.run(accuracy)
    
    # 保存模型
    save_path = saver.save(sess, model_path)
        