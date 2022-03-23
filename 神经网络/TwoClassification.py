# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:20:48 2022
linear logistic regression to deal with two classification
@author: Xinnze
"""
import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

tf.disable_v2_behavior()


# 1.生成样本集

# regression=True, 表示使用非onehot编码的标签
def generate(sample_size, mean, cov, diff, regression):
    num_classes = 2
    samples_pre_class = int(sample_size / 2)

    # 生成多元正态分布矩阵 矩阵大小为 samples_pre_class * len(mean)
    X0 = np.random.multivariate_normal(mean, cov, samples_pre_class)
    Y0 = np.zeros(samples_pre_class)

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean + d, cov, samples_pre_class)
        Y1 = (ci + 1) * np.ones(samples_pre_class)

        X0 = np.concatenate((X0, X1))  # 矩阵拼接  axis默认为0
        Y0 = np.concatenate((Y0, Y1))

    # one_hot编码 将 0 转成 1 0
    if regression is False:
    # 两个写法
# =============================================================================
#         Y0_1 = []
#         for i in range(len(Y0)):
#             Y0_1.append([1 if Y0[i] == class_number else 0 for class_number in range(num_classes)])
#         Y0 = np.asarray(Y0_1, np.float32)
# =============================================================================
        Y0 = np.asarray([[1 if Y0[i] == class_number else 0 for class_number in range(num_classes)] for i in range(len(Y0))],
                        np.float32)
        
    X, Y = shuffle(X0, Y0)  # 打乱多组数据

    return X, Y


# 定义随机数的种子值 保证每一次代码运行生成的随机数时一样的
np.random.seed(10)
num_classes = 2
mean = np.random.randn(num_classes)
# print(mean)
cov = np.eye(num_classes)
X, Y = generate(1000, mean, cov, [3.0], True)
# print(X, Y)
# =============================================================================
# colors = ['r' if l == 0 else 'b' for l in Y[: ]]
# plt.scatter(X[:, 0], X[:, 1], c=colors)
# plt.xlabel("Scaled age (in yrs)")
# plt.ylabel("Tumor size (in cm)")
# plt.show()
# =============================================================================

input_dim = 2
lab_dim = 1


# 2 构建网络结构
input_features = tf.placeholder(tf.float32, [None, input_dim])
input_labels = tf.placeholder(tf.float32, [None, lab_dim])

# 定义学习参数
W = tf.Variable(tf.random_normal([input_dim, lab_dim]), name="weight")
b = tf.Variable(tf.zeros([lab_dim]), name="bias")  # b是一个标量

output = tf.nn.sigmoid(tf.matmul(input_features, W) + b)  # 输出
cross_entropy = -(input_labels * tf.log(output) + (1 - input_labels) * tf.log(1 - output))  # 交叉熵

ser = tf.square(input_labels - output)
err = tf.reduce_mean(ser)  #平方差函数 估计模型的错误率

loss = tf.reduce_mean(cross_entropy)  # 损失函数还是使用交叉熵
optimizer = tf.train.AdamOptimizer(0.04)  # Adamoptimizer 尽量用这个 因为收敛速度快 会动态调节梯度
train = optimizer.minimize(loss)

# 3 设置参数进行迭代经验
maxEpochs = 50
minbatchsize = 25

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(b.shape)
    
    # 向目标传入数据
    for epoch in range(maxEpochs):
        avgerr = 0  # 累计错误
        avgcost = 0  # 平均loss值
        for i in range(np.int32(len(Y) / minbatchsize)):
            x1 = X[i * minbatchsize: (i + 1) * minbatchsize, : ]
            y1 = np.reshape(Y[i * minbatchsize: (i + 1)* minbatchsize], [-1, 1])
            
            _, lossval, outputval, errval = sess.run([train, loss, output, err], 
                                                     feed_dict={input_features: x1, input_labels: y1})
            avgerr += errval / np.int32(len(Y) / minbatchsize)
            avgcost += lossval / np.int32(len(Y) / minbatchsize)
        print("Epoch: ", "%04d"  % (epoch + 1), "cost= ", "{:.9f}".format(avgcost), "err = ", avgerr)
        # print('\n', sess.run(b))
    print("Finished! ")

    # 数据可视化
    train_X, train_Y = generate(100, mean, cov, [3.0], regression=True)
    colors = ['r' if l == 0 else 'b' for l in Y[: ]]
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    x = np.linspace(-1 ,8, 200)
    y = -x * (sess.run(W)[0] / sess.run(W)[1]) - sess.run(b) / sess.run(W)[1]
    plt.plot(x, y, label="Fitted line")
    plt.legend()
    plt.show()
# 可见是 线性可分 的模型
