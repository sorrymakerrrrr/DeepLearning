# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 19:26:10 2022
Solve XOR Overfitting
1.regularization  L1, L2 ...
@author: Xinnze
"""
import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, colorConverter

tf.disable_v2_behavior()

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


# 1. 构建异或数据集
np.random.seed(10)

input_dim = 2
num_class = 4
mean = np.random.randn(input_dim)
cov = np.eye(input_dim)
X, Y = generate(320, num_class, mean, cov, [[3.0, 0], [3.0, 3.0], [0, 3.0]], True)
Y = Y % 2 # 取余数
Y = np.reshape(Y, [-1, 1])

# =============================================================================
# # 数据分为两类，其中左下和右上是一类(红色用+表示) 左上和右下是一类(蓝色用.表示)
# xr = []
# xb = []
# for (l, k) in zip(Y[: ], X[: ]):
#     if l == 0.0:
#         xr.append([k[0], k[1]])
#     else:
#         xb.append([k[0], k[1]])
#         
# xr = np.array(xr)
# xb = np.array(xb)
# 
# plt.scatter(xr[:, 0], xr[:, 1], c='r', marker='+')
# plt.scatter(xb[:, 0], xb[:, 1], c='b', marker='o')
# plt.show()
# =============================================================================

# 2.修改定义网络模型
n_label = 1 
n_hidden = 200   # 增加节点或者增加层，让模型具有更好的拟合性
x = tf.placeholder(tf.float32, [None, input_dim])
y = tf.placeholder(tf.float32, [None, n_label])

learning_rate = 1e-4
W = {
     # 隐藏层有n_hidden个个数的神经元，输入有input_dim个特征，所以就有input_dim*n_hidden个权重
     'h1': tf.Variable(tf.truncated_normal([input_dim, n_hidden], stddev=0.1)),
     'h2': tf.Variable(tf.truncated_normal([n_hidden, n_label], stddev=0.1))
     }

b = {
     'h1': tf.Variable(tf.zeros([n_hidden])),
     'h2': tf.Variable(tf.zeros([n_label]))
     }

layer_1 = tf.nn.relu(tf.matmul(x, W['h1']) + b['h1'])
y_pred = tf.nn.tanh(tf.matmul(layer_1, W['h2']) + b['h2'])

# loss = tf.reduce_mean((y - y_pred) ** 2)

# 用l2正则化来改善过拟合
reg = 0.01
loss = tf.reduce_mean((y - y_pred) ** 2) + reg * (tf.nn.l2_loss(W['h1']) + tf.nn.l2_loss(W['h2']))

# =============================================================================
# # 用l1正则化来改善过拟合
# reg = 0.01
# loss = tf.reduce_mean((y - y_pred) ** 2) + reg * (tf.reduce_sum(tf.abs(W['h1'])) + tf.reduce_sum(tf.abs(W['h2'])))
# =============================================================================

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1, 10001):
        _, lossval = sess.run([train_step, loss], feed_dict={x: X, y: Y})
        
        if epoch % 100 == 0:
            print("Epoch: ", "%04d" % epoch, "loss = ", "{:.9f}".format(lossval))
    print("Finished! ")
    # print(sess.run(y_pred, feed_dict={x: X, y: Y}))
    
# =============================================================================
# # 3.添加可视化
#     xTrain, yTrain = generate(120, num_class, mean, cov, [[3.0, 0], [3.0, 3.0], [0, 3.0]], True)
#     yTrain = np.reshape(yTrain % 2, [-1, 1])
#     xr = []
#     xb = []
#     for (l, k) in zip(yTrain[: ], xTrain[: ]):
#         if l == 0.0:
#             xr.append([k[0], k[1]])
#         else:
#             xb.append([k[0], k[1]])
#             
#     xr = np.array(xr)
#     xb = np.array(xb)
#     plt.scatter(xr[:, 0], xr[:, 1], c='r', marker='+')
#     plt.scatter(xb[:, 0], xb[:, 1], c='b', marker='o')
#     
#     print("loss:\n", sess.run(loss, feed_dict={x: xTrain, y: yTrain}))
#     
#     nb_of_xs = 200 
#     xs1 = np.linspace(-1, 8, num=nb_of_xs)
#     xs2 = np.linspace(-1, 8, num=nb_of_xs)
#     xx, yy = np.meshgrid(xs1, xs2)
#     classification_plane = np.zeros((nb_of_xs, nb_of_xs))
#     
#     for i in range(nb_of_xs):
#         for j in range(nb_of_xs):
#             classification_plane[i, j] = sess.run(y_pred, feed_dict={x: [[xx[i, j], yy[i, j]]]})
#             classification_plane[i, j] = round(classification_plane[i, j])
#     
#     cmap = ListedColormap(
#         colorConverter.to_rgba('y', alpha=0.3),
#         colorConverter.to_rgba('b', alpha=0.3)
#         )
#     plt.xlim(-1, 8)
#     plt.ylim(-1, 8)
#     plt.contourf(xx, yy, classification_plane, cmap=cmap)
#     plt.show()
# =============================================================================
    
# 4. 验证过拟合
    xTrain1, yTrain1 = generate(20, num_class, mean, cov, [[3.0, 0], [3.0, 3.0], [0, 3.0]], True)
    yTrain1 = np.reshape(yTrain1 % 2, [-1, 1])
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
    
    print("loss:\n", sess.run(loss, feed_dict={x: xTrain1, y: yTrain1}))
    
    nb_of_xs = 200
    xs1 = np.linspace(-1, 8, num=nb_of_xs)
    xs2 = np.linspace(-1, 8, num=nb_of_xs)
    xx, yy = np.meshgrid(xs1, xs2)
    classification_plane = np.zeros((nb_of_xs, nb_of_xs))
    
    for i in range(nb_of_xs):
        for j in range(nb_of_xs):
            classification_plane[i, j] = sess.run(y_pred, feed_dict={x: [[xx[i, j], yy[i, j]]]})
            classification_plane[i, j] = round(classification_plane[i, j])
    
    cmap = ListedColormap(
        colorConverter.to_rgba('r', alpha=0.3),
        colorConverter.to_rgba('b', alpha=0.3)
        )
    plt.xlim(-1, 8)
    plt.ylim(-1, 8)
    plt.contourf(xx, yy, classification_plane, cmap=cmap)
    plt.show()
