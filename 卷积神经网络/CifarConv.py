# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 17:14:22 2022
使用三个卷积层的同卷积操作 滤波器为5*5 每一个卷积层后面都会跟一个步长为2*2的池化层，滤波器为2*2
2层的卷积加池化后是输出为10个通道的卷积层，然后对这10个feature map进行全局平均池化，得到10个特征
再对十个特征进行softmax计算，结果代表最终分类
@author: Xinnze
"""


import tensorflow.compat.v1 as tf 
from cifar10 import cifar10_input 
import numpy as np 


batch_size = 128 
data_dir = 'cifar-10-batches-bin'

print('begin')

images_train, labels_train = cifar10_input.inputs(eval_data=False, data_dir=data_dir, batch_size=batch_size)
images_test, labels_tests = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)
print('begin data')

# 权重w定义
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 滤波器为5*5 卷积进行同卷积操作 即步长为1，padding = "SAME" 
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    

# 步长为2，padding为'SAME' 即将卷积缩小一半
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 放在最后一层 取均值 步长为最终生成的尺寸
def avg_pool_6x6(x):
    return tf.nn.avg_pool(x, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME')


# 定义占位符
x = tf.placeholder(tf.float32, [None, 24, 24, 3])  # 此时cifar data的shape为 24*24*3 而不是1*24*24*3
y = tf.placeholder(tf.float32, [None, 10])  # 0-9数字分类

# 三通道输入，输出64个feature map 每一个feature map都是三个通道卷积生成的feature map对应位置相加得到
x_image = tf.reshape(x, [-1, 24, 24, 3])  # 好像不要这一步也可以 conv2d(x_image, W_conv1) 写成 conv2d(x, W_conv1)
W_conv1 = weight_variable([5, 5, 3, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_conv3 = weight_variable([5, 5, 64, 10])
b_conv3 = bias_variable([10])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

nt_hpool3 = avg_pool_6x6(h_conv3)  # shape 为(-1, 1, 1, 10)
nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
# nt_hpool3_flat = nt_hpool3

y_conv = tf.nn.softmax(nt_hpool3_flat)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y* tf.log(y_conv), 1))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

y_pred = tf.argmax(y_conv, 1)
correct_prediction = tf.equal(y_pred, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess = tf.Session()

sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)  # 启动队列

for i in range(15000):
    image_batch, label_batch = sess.run([images_train, labels_train])
    label_b = np.eye(10)[label_batch]  # 将label_batch转成 独热编码
    
    train_step.run(feed_dict={x: image_batch, y: label_b}, session=sess)
    
    if i % 200 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: image_batch, y: label_b},session=sess)
        loss = cross_entropy.eval(feed_dict={x: image_batch, y: label_b}, session=sess)
        
        print('step %d, loss %g, train_accuary %g' % (i, loss, train_accuracy))
        # print(nt_hpool3.shape, '\n', nt_hpool3_flat.shape)
        # print(x.shape, x_image.shape)

print('Finish!')
        
