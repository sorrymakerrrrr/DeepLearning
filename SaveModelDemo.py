# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 13:50:49 2022

@author: Xinnze
"""
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

tf.disable_v2_behavior()
# 计算loss
def moving_average(a, w=10):
    if len(a) < w:
        return a[: ]
    return [val if idx < w else sum(a[idx - w: idx]) / w 
            for idx, val in enumerate(a)]


# 模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
tf.reset_default_graph()  # 清除默认图形堆栈并重置全局默认图形

# 初始化
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
paradict = {
    'W': tf.Variable(tf.random_normal([1]), name="weight"),
    'b': tf.Variable(tf.zeros([1]), name="bias")
    }
z = (paradict['W'] * X) + paradict['b']
cost = tf.reduce_mean(tf.square(Y - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
train_epoch = 20
display_step = 2

saver = tf.train.Saver() # 生成saver
# =============================================================================
# saver = tf.train.Saver({"weight": w, "bias": b})  # 指定存储变量名字与变量的对应关系
# saver = tf.train.Saver([W, b])  # 放到一个list里面
# saver = tf.train.Saver({v.op.name: v for v in [paradict['W'], paradict['b']]})
# =============================================================================
savedir = "E:\\机器学习_myself\\深度学习\\数据存储\\线性回归\\"  # 生成模型的路径

# 启动会话
with tf.Session() as sess:
    sess.run(init)
    plotdata={"batch_size":[], "loss":[]}
    for epoch in range(train_epoch):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: x, Y: y})
            print("Epoch: ", epoch + 1, "cost: ", loss, 
                  "W: ", sess.run(paradict['W']), "b: ", sess.run(paradict['b']))
            if not (loss == "NA"):
                plotdata["batch_size"].append(epoch)
                plotdata["loss"].append(loss)
    saver.save(sess, savedir + "linemodel.ckpt")  # 保存模型
    print("Finished!")
    print("cost: ", sess.run(cost, feed_dict={X: x, Y: y}), 
          "W: ", sess.run(paradict['W']), "b: ", sess.run(paradict['b']))

# 可视化
    plt.figure(1)
    plt.subplot(211)
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(paradict['W']) * train_X + sess.run(paradict['b']),
             label='Fittedline')
    plt.legend()
    plt.subplot(212)
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.plot(plotdata["batch_size"], plotdata["avgloss"], color='black', linestyle='--')
    plt.xlabel("Minibatch number")
    plt.ylabel("Loss")
    plt.title("Minibatch run vs. Training loss")
    plt.show()
    
    # 使用模型
    print("x: 15, y_pridict/z: ", sess.run(z, feed_dict={X: 15}))
        