# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:37:04 2022
Solve XOR Overfitting 
2.dropout
3.data agumentation
@author: Xinnze
"""
# =============================================================================
# dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
# X: 输入的模型节点
# keep_prob: 保持率 如果为1，代表全部进行学习；如果为0.8，代表只让80%的节点进行学习
# noise_shape: 代表指定x中，哪些维度可以使用dropout技术。None时所有的维度使用dropout技术
#              x.shape = [n, len, w, ch] 使用noise_shape=[n, 1, 1, ch]，表明会对x中第二维度len与第三维度w进行dropout
# seed:随机选取节点过程中随机数的种子值
# =============================================================================
import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, colorConverter

tf.disable_v2_behavior()
# tf.reset_default_graph()

def generate(samples_size, num_class, mean, cov, diff, regression):
    samples_per_size = int(samples_size / num_class)
    x0 = np.random.multivariate_normal(mean, cov, samples_per_size)
    y0 = np.zeros(samples_per_size)
    
    for ci, d in enumerate(diff):
        x1 = np.random.multivariate_normal(mean + d, cov, samples_per_size)
        y1 = (ci + 1) * np.ones(samples_per_size)
        
        x0 = np.concatenate([x0, x1])
        y0 = np.concatenate([y0, y1])
    
    if regression is False:
        y0 = [[1 if y0[i] == class_number else 0 for class_number in range(num_class)] for i in range(len(y0))]
        y0 = np.asarray(y0)
        
    X, Y = shuffle(x0, y0)
    return X, Y


# 定义学习参数
# np.random.seed(10)
num_classes = 4
n_input = 2  # 输入层节点个数
n_label = 1
n_hidden = 200  # 隐藏层节点个数
mean = np.random.randn(n_input)
cov = np.eye(n_input)

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_label])

learning_rate = 0.01

weights = {
    # tf.truncated_normal(shape, mean, stddev)
    # 截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1)),
    'h2': tf.Variable(tf.truncated_normal([n_hidden, n_label], stddev=0.1))
    }
biases = {
    'h1': tf.Variable(tf.zeros([n_hidden])),
    'h2': tf.Variable(tf.zeros([n_label]))
    }

# 定义网络模型
layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['h1']))

keep_prob = tf.placeholder("float")
layer_1_drop = tf.nn.dropout(layer_1, keep_prob)


layer2 = tf.add(tf.matmul(layer_1_drop, weights['h2']), biases['h2'])
y_pred = tf.maximum(layer2, 0.01 * layer2)  # Leaky relus激活函数

loss = tf.reduce_mean((y - y_pred) ** 2)
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 在最后几次中出现了抖动的现象，表明后期的学习率有点太大了  使用退化学习率
global_step = tf.Variable(0, trainable=False)
decaylearning_rate = tf.train.exponential_decay(learning_rate, global_step, 1000, 0.9)
train_step = tf.train.AdamOptimizer(decaylearning_rate).minimize(loss, global_step=global_step)
add_global = global_step.assign_add(1)  # 定义一个op,令global_step加1完成计步

# 启动session会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        
        X, Y = generate(1000, num_classes, mean, cov, [[3.0, 0], [3.0, 3.0], [0, 3.0]], True)
        Y = np.reshape(Y % 2,[-1, 1])
        
        _, _, loss_val = sess.run([add_global, train_step, loss], feed_dict={x: X, y: Y, keep_prob: 0.6})
        
        if i % 1000 == 0:
            print("step: ", i, "Current loss:", loss_val)
          
            
    # 可视化
    xTrain1, yTrain1 = generate(200, num_classes, mean, cov, [[3.0, 0], [3.0, 3.0], [0, 3.0]], True)
    yTrain1 = np.reshape(yTrain1 % 2, [-1, 1])
    print("loss:\n", sess.run(loss, feed_dict={x: xTrain1, y: yTrain1, keep_prob: 1}))
    xr = []
    xb = []
    for (l, k) in zip(yTrain1[: ], xTrain1[: ]):
        if l == 0.0:
            xr.append([k[0], k[1]])
        else:
            xb.append([k[0], k[1]])
            
    xr = np.array(xr)
    xb = np.array(xb)
    plt.scatter(xr[:, 0], xr[:, 1], c='r', marker='+')
    plt.scatter(xb[:, 0], xb[:, 1], c='b', marker='o')
    
    nb_of_xs = 200
    xs1 = np.linspace(-1, 8, num=nb_of_xs)
    xs2 = np.linspace(-1, 8, num=nb_of_xs)
    xx, yy = np.meshgrid(xs1, xs2)
    classification_plane = np.zeros((nb_of_xs, nb_of_xs))
    
    for i in range(nb_of_xs):
        for j in range(nb_of_xs):
            classification_plane[i, j] = sess.run(y_pred, feed_dict={x: [[xx[i, j], yy[i, j]]], keep_prob: 1})
            classification_plane[i, j] = round(classification_plane[i, j])
    
    cmap = ListedColormap(
        colorConverter.to_rgba('r', alpha=0.3),
        colorConverter.to_rgba('b', alpha=0.3)
        )
    plt.xlim(-1, 8)
    plt.ylim(-1, 8)
    plt.contourf(xx, yy, classification_plane, cmap=cmap)
    plt.show()
        
