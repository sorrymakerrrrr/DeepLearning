# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 09:47:26 2022
linear logistic regression to deal with many classification
@author: Xinnze
"""

import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, colorConverter

tf.disable_v2_behavior() 


# 1.生成yangbenji
def generate(sample_size, num_class, cov, mean, diff, regression):
    sample_pre_class = int(sample_size / num_class)
    
    X0 = np.random.multivariate_normal(mean, cov, sample_pre_class)
    Y0 = np.zeros(sample_pre_class)
    
    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean + d, cov, sample_pre_class)
        Y1 = (ci + 1) * np.ones(sample_pre_class)
        
        X0 = np.concatenate((X0, X1))  # 拼接的矩阵要加括号
        Y0 = np.concatenate((Y0, Y1))
    
    
    if regression == False:  # 此时将Y变成独热编码
        Y0 = [[1 if Y0[i] == class_number else 0 for class_number in range(num_class)] for i in range(len(Y0))]
        Y0 = np.asarray(Y0, np.float32)
    
    X, Y = shuffle(X0, Y0)  # 打乱顺序 X0, Y0 一一对应
    
    return X, Y

np.random.seed(10)

input_dim = 2
num_classes = 3
mean = np.random.randn(input_dim)
cov = np.eye(input_dim)
X, Y = generate(2000, num_classes, cov, mean, [[3.0], [3.0, 0]], False)
# print(X, Y)

# =============================================================================
# aa = [np.argmax(l) for l in Y]
# colors = ['r' if l == 0 else 'b' if l == 1 else 'y' for l in aa]
# plt.scatter(X[:, 0], X[:, 1], c=colors)
# plt.show()
# =============================================================================

lab_dim = num_classes

# 2. 构建网络结构
# 定义占位符
input_features = tf.placeholder(tf.float32, [None, input_dim])
input_labels = tf.placeholder(tf.float32, [None, lab_dim])
# 定义学习参数
W = tf.Variable(tf.random_normal([input_dim, lab_dim]), name="weight")
b = tf.Variable(tf.random_normal([lab_dim]), name="bias")

output = tf.matmul(input_features, W) + b
z = tf.nn.sigmoid(output)

a1 = tf.argmax(z, axis=1)  # 按行找出 最大索引，生成数组
b1 = tf.argmax(input_labels, axis=1)
err = tf.count_nonzero(a1 - b1)  #两个数组相减，不为0的就是错误的个数

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=output)  # 交叉熵
loss = tf.reduce_mean(cross_entropy)  # 对一个批次的交叉熵取均值

optimizer = tf.train.AdamOptimizer(0.04)
train = optimizer.minimize(loss)

# 3. 设置参数进行训练
maxEpochs = 50
minibatchSize = 25

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(maxEpochs):
        sumerr = 0
        avgcost = 0
        for i in range(np.int32(len(Y) / minibatchSize)):
            x1 = X[minibatchSize * i: minibatchSize * (i + 1), :]
            y1 = Y[minibatchSize * i: minibatchSize * (i + 1), :]
            
            _, lossval, outputval, errval = sess.run([train, loss, z, err], 
                                                     feed_dict={input_features: x1, input_labels: y1})
            
            sumerr += errval / minibatchSize
            avgcost += lossval / (np.int32(len(Y) / minibatchSize))  # 平均交叉熵
        
        print("Epoch: ", "%04d" % (epoch + 1), "cost = ", "{:.9f}".format(avgcost),
              "err = ", sumerr / np.int32(len(Y) / minibatchSize))
    print("Finished! ")
    
    # 数据可视化
    train_X, train_Y = generate(200, 3, cov, mean, [[3.0], [3.0, 0]], regression=False)
    aa = [np.argmax(l) for l in train_Y]
    colors = ['r' if l == 0 else 'b' if l == 1 else 'y' for l in aa]
    plt.scatter(train_X[:, 0], train_X[:, 1], c=colors)
    
    x = np.linspace(-1, 8, 200)
    
    y = -x * (sess.run(W)[0][0] / sess.run(W)[1][0]) - (sess.run(b)[0]) / sess.run(W)[1][0]
    plt.plot(x, y, label='first line', lw=3)
    
    y = -x * (sess.run(W)[0][1] / sess.run(W)[1][1]) - (sess.run(b)[1]) / sess.run(W)[1][1]
    plt.plot(x, y, label='second line', lw=2)
    
    y = -x * (sess.run(W)[0][2] / sess.run(W)[1][2]) - (sess.run(b)[2]) / sess.run(W)[1][2]
    plt.plot(x, y, label='third line', lw=1)
    # plt.legend()
    # plt.show()
    
    print(sess.run(W), '\n', sess.run(b))
    
    # 5. 模型可视化  显示更加直观
    nb_of_xs = 200
    xs1 = np.linspace(-1, 8, num=nb_of_xs)
    xs2 = np.linspace(-1, 8, num=nb_of_xs)
    xx, yy = np.meshgrid(xs1, xs2)  # 创建网格
    # 初始化与填充
    classification_plane = np.zeros((nb_of_xs, nb_of_xs))
    
    for i in range(nb_of_xs):
        for j in range(nb_of_xs):
            # a1为z的最大值索引
            classification_plane[i, j] = sess.run(a1, feed_dict={input_features:[[xx[i, j], yy[i, j]]]})
      
    # 创建colormap用于显示
    cmap = ListedColormap([colorConverter.to_rgba('r', alpha=0.30),
                           colorConverter.to_rgba('b', alpha=0.30),
                           colorConverter.to_rgba('y', alpha=0.30)
                          ])
    # 图示各个样本边界
    # 绘制等高线图  contoourf会填充轮廓 contour不会填充轮廓
    # coutour(X, Y, Z, **kwargs) 
    # 当 X,Y,Z 都是 2 维数组时，它们的形状必须相同。如果都是 1 维数组时，len(X)是 Z 的列数，而 len(Y) 是 Z 中的行数。（例如，经由创建np.meshgrid()）
    plt.contourf(xx, yy, classification_plane, cmap=cmap)
    plt.xlim(-1, 8)
    plt.ylim(-1, 8)
    plt.legend()
    plt.show()
    