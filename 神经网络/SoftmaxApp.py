# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:44:11 2022

@author: Xinnze
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 6.5.1 交叉熵实验
labels = [[0, 0, 1], [0, 1, 0]]
logits = [[2, 0.5, 6], [0.1, 0, 3]]

logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)

result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # logits与labels的softmax交叉熵
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)
result3 = -tf.reduce_sum(labels * tf.log(logits_scaled), 1)  # 自建公式logits与labels的softmax交叉熵
result4 = -tf.reduce_sum(labels * tf.log(logits_scaled2), 1)

with tf.Session() as sess:
    # print(tf.Graph())
    # print(sess.run(labels*tf.constant(logits)))
    print('logit_scaled: ', sess.run(logits_scaled))
    print('logit_scaled2: ', sess.run(logits_scaled2))
    print("rel1 =", sess.run(result1), '\n')  # 可见传入softmax_cross_entropy_with_logits的值是不需要进行softmax的
    print("rel2 =", sess.run(result2), '\n')  # 将进行softmax后的值再传入函数，结果会出错，相当于进行了两次softmax转换
    print("rel3 =", sess.run(result3), '\n')
    print("rel4 =", sess.run(result4), '\n')

# 6.5.2 one_hot实验
# 对非one_hot编码为 标签 的数据进行交叉熵的计算，比较其与one_hot编码的交叉熵之间的差别
labels = [[0.4, 0.1, 0.5], [0.3, 0.6, 0.1]]  # 标签总概率为1
result5 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

with tf.Session() as sess:
    # 对于正确分类的交叉熵和错误分类的交叉熵，两者的结果没有标准one_hot这么明显
    print("rel5 =", sess.run(result5), '\n')

# 6.5.3 sparse 标签
labels2 = [2, 1]  # 表明labels总共分3个类0、1、2 [2, 1]等价于one_hot编码中的001与010
result6 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels2, logits=logits)
with tf.Session() as sess:
    print("rel6 = ", sess.run(result6))  # 结果与前面的rel1完全一样

# 6.5.4 计算loss值
loss = tf.reduce_mean(result1)
loss2 = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits_scaled), 1))
loss3 = tf.reduce_mean(result3)
# loss1 loss2 loss3得到的值一样
with tf.Session() as sess2:
    # print(tf.Graph())
    print('loss = ', sess2.run(loss), '\n')
    print('loss2 = ', sess2.run(loss2), '\n')
    print('loss3 = ', sess2.run(loss3), '\n')
    
