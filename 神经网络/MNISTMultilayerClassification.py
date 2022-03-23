# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:39:52 2022
MNIST Multi-layer Classification
@author: Xinnze
"""

import tensorflow.compat.v1 as tf
from tensorflow.core.example.tutorials.mnist import input_data

# 读数据集
tf.disable_v2_behavior()
tf.reset_default_graph()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print(mnist.train.images[1])

# 1.定义网络参数
learning_rate = 0.001
training_epochs = 25
batch_size = 100 
display_step = 1 

#设置网络模型参数
n_hidden_1 = 256  # 第一个隐藏层节点个数
n_hidden_2 = 256  # 第二个隐藏层节点个数
n_input = 784  # MNIST共784(28 * 28)维
n_classes = 10  # MNIST共10个类别

# 2. 定义网络结构
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# 创建model
def multilayer_perception(x, weights, biases):
    # 第一层隐藏层
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # 第二层隐藏层
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # 输出层
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return out_layer

# 学习参数
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])), 
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])), 
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
    }

# 输出值
pred = multilayer_perception(x, weights, biases)

# 定义loss和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, lossval = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            
            avg_cost += lossval / total_batch
        
        if (epoch + 1) % display_step == 0:
            print("Epoch: ", '%04d' % (epoch + 1), "cost = ", "{:.9f}".format(avg_cost))
    print('Finished! ')
    
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))  # axis=1 取每一行的最大索引
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
