# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 18:28:39 2022

@author: Xinnze
"""

import tensorflow.compat.v1 as tf
from cifar10 import cifar10_input
import numpy as np
from Unpooling import max_pool_with_argmax, unpool


tf.Graph().as_default()
tf.disable_v2_behavior()

batch_size = 128
data_dir = 'cifar-10-batches-bin'

print('begin')

images_train, labels_train = cifar10_input.inputs(eval_data=False, data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)
print('begin data')


# 权重w定义
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)  # 是一个变量


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积进行同卷积操作 即步长为1 padding="SAME"
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 最后一层全局池化层
def avg_pool_6x6(x):
    return tf.nn.avg_pool(x, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME')


# 定义占位符
x = tf.placeholder(tf.float32, shape=[128, 24, 24, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])

# 定义网络结构
x_image = tf.reshape(x, [-1, 24, 24, 3])

W_conv1 = weight_variable([5, 5, 3, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1, mask1 = max_pool_with_argmax(h_conv1, stride=2)  # 改成带mask返回值的max_pool_with_argmax函数

W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2, mask2 = max_pool_with_argmax(h_conv2, stride=2)  # (128, 6, 6, 64)
# 打印形状 是在组建网络结构时常用的一种调试方法
# 反卷积和反池化对形状都很敏感 这种方法可以看出当前我们输入的是什么形状
print(h_pool2.shape)

W_conv3 = weight_variable([5, 5, 64, 10])
b_conv3 = bias_variable([10])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
nt_hpool3 = avg_pool_6x6(h_conv3)
nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])

y_conv = tf.nn.softmax(nt_hpool3_flat)
# 加正则化项
cross_entropy = -tf.reduce_sum(y * tf.log(y_conv)) + tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_conv3) 

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 以第二层池化输出的变量h_pool2为开始部分，沿着h_pool2生成方向反向操作一层一层推导，直到生成原始图t1_x_image
# 每个卷积之后都会加上一个偏置项b，在反卷积过程中就要将b减去。
# 由于relu函数基本上是恒等变化(除了小于0的部分)，所以不需要做可逆操作，可以直接略去

# 第二层的卷积还原
t_conv2 = unpool(h_pool2, mask2, 2)
# strides=[1, 1, 1, 1], padding='SAME'同conv2d的参数
t_pool1 = tf.nn.conv2d_transpose(t_conv2, W_conv2, h_pool1.shape, strides=[1, 1, 1, 1], padding='SAME')
print(t_conv2.shape, h_pool1.shape, t_pool1.shape)  # h_pool1.shape, t_pool1.shape是一致的

t_conv1 = unpool(t_pool1, mask1, 2)
t_x_image = tf.nn.conv2d_transpose(t_conv1, W_conv1, x_image.shape, strides=[1, 1, 1, 1], padding='SAME')



# 第一层卷积还原 从h_pool1开始
t1_conv1 = unpool(h_pool1, mask1, 2)
t1_x_image = tf.nn.conv2d_transpose(t1_conv1, W_conv1, x_image.shape, strides=[1, 1, 1, 1])  # padding='SAME


# 生成最终图像
stitched_decodings = tf.concat([x_image, t1_x_image, t_x_image], axis=2)
decoding_summary_op = tf.summary.image('source/cifar', stitched_decodings)

# session中写入log
# 在session中建立一个summary_writer
# 然后在代码结尾处通过运行tf.summary.image操作 使用summary_writer将得出的结果写入log
sess = tf.Session()
sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter('./log/', sess.graph)
tf.train.start_queue_runners(sess=sess)  # 启动队列

for i in range(15000):
    image_batch, label_batch = sess.run([images_train, labels_train])
    label_b = np.eye(10)[label_batch]  # 转成one_hot 编码
    
    train_step.run(feed_dict={x: image_batch, y: label_b}, session=sess)
    
    if i % 200 == 0:
        loss = cross_entropy.eval(feed_dict={x: image_batch, y: label_b}, session=sess)
        train_accuracy = accuracy.eval(feed_dict={x: image_batch, y: label_b}, session=sess)
        print('Epoch: %d, loss= %g, train_accuracy= %g' % (i, loss, train_accuracy))
    
    decoding_summary = sess.run(decoding_summary_op, feed_dict={x:image_batch, y: label_b})
    summary_writer.add_summary(decoding_summary)

print('Finished! ')

        
