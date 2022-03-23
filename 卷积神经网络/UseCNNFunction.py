# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:47:17 2022
Use CNN function
@author: Xinnze
"""
"""
tf.nn.conv2d(input, fliter, strides, padding, use_cudnn_on_gpu=None, name=None)
input: [batch, in_height, in_width, in_channels]  type:Tensor  要求类型float32, float64
fliter: [fliter_height, fliter_width, in_channels, out_channels]  type:Tensor
         卷积核高度      卷积核宽度     图像通道数    卷积核个数
padding: "SAME"  "VALUE"
strides: 卷积时在图像每一维的步长，是一个一维的向量，[1, strides, strides, 1]，第一位和最后一位固定必须是1

"""

import tensorflow.compat.v1 as tf 

tf.disable_v2_behavior()

# 1. 定义3个输入变量用来模拟输入图片 分别是5*5大小1通道的矩阵 5*5大小2通道的矩阵 4*4大小1通道的矩阵，并将里面的值通通赋为1

# [batch, in_height, in_width, in_channels] [训练时一个批次的图片数量，图片高度，图片宽度，图像通道数]
input1 = tf.Variable(tf.constant(1.0, shape=[1, 5, 5, 1]))
input2 = tf.Variable(tf.constant(1.0, shape=[1, 5, 5, 2]))
input3 = tf.Variable(tf.constant(1.0, shape=[1, 4, 4, 1]))

# 2. 定义卷积核变量
# 定义5个卷积核，每个卷积核都是2*2的矩阵，只有输入，输出的通道数有差别，
# 分别为1ch输入、1ch输出; 1ch输入、2ch输出; 1ch输入、3ch输出; 2ch输入、2ch输出; 2ch输入、1ch输出
# [fliter_height, fliter_width, in_channels, out_channels] [卷积核高度 卷积核宽度 图像通道数 卷积核个数]
fliter1 = tf.Variable(tf.constant([-1.0, 0, 0, -1], shape=[2, 2, 1, 1]))

fliter2 = tf.Variable(tf.constant([-1.0, 0, 0, -1, 
                                   -1.0, 0, 0, -1], shape=[2, 2, 1, 2]))

fliter3 = tf.Variable(tf.constant([-1.0, 0, 0, -1, 
                                   -1.0, 0, 0, -1, 
                                   -1.0, 0, 0, -1], shape=[2, 2, 1, 3]))

fliter4 = tf.Variable(tf.constant([-1.0, 0, 0, -1,
                                   -1.0, 0, 0, -1,
                                   -1.0, 0, 0, -1,
                                   -1.0, 0, 0, -1], shape=[2, 2, 2, 2]))

fliter5 = tf.Variable(tf.constant([-1.0, 0, 0, -1,
                                   -1.0, 0, 0, -1], shape=[2, 2, 2, 1]))


# 3. 定义卷积操作
# 将步骤1与步骤2的卷积核组合起来。建立8个卷积操作  观察生成内容与课本的讲述是否一致
# padding的值为”VAILD“表示边缘不填充; 值为"SAME"时表示填充卷积核到图像边缘

# 演示padding补0的情况 一个通道输入，生成一个feature map
op1 = tf.nn.conv2d(input1, fliter1, strides=[1, 2, 2, 1], padding='SAME')

# 对于padding的不同而不同
vop1 = tf.nn.conv2d(input1, fliter1, strides=[1, 2, 2, 1], padding='VALID')

# 一个通道输入，生成两个feature map
op2 = tf.nn.conv2d(input1, fliter2, strides=[1, 2, 2, 1], padding='SAME')

# 一个通道输入，生成三个feature map
op3 = tf.nn.conv2d(input1, fliter3, strides=[1, 2, 2, 1], padding='SAME')

# 两个通道输入，生成两个feature map
op4 = tf.nn.conv2d(input2, fliter4, strides=[1, 2, 2, 1], padding='SAME')

# 两个通道输入，生成一个feature map
op5 = tf.nn.conv2d(input2, fliter5, strides=[1, 2, 2, 1], padding='SAME')

# 4*4 与padding无关
op6 = tf.nn.conv2d(input3, fliter1, strides=[1, 2, 2, 1], padding='SAME')

vop6 = tf.nn.conv2d(input3, fliter1, strides=[1, 2, 2, 1], padding='VALID')

# print(op1.shape)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("op1:\n", sess.run([op1, fliter1]))  # 1-1 后面补0
    print('------------------------------------------------------------------')
    
    print("op2:\n", sess.run([op2, fliter2]))  # 1-2  多卷积核，按列取(重点理解按列取)
    print("op3:\n", sess.run([op3, fliter3]))  # 1-3  一个输入，三个输出
    print('------------------------------------------------------------------')
    
    print("op4:\n", sess.run([op4, fliter4]))  # 2-2 通道叠加
    print("op5:\n", sess.run([op5, fliter5]))  # 2-1 两个输入，一个输出
    print('------------------------------------------------------------------')

    print("op1:\n", sess.run([op1, fliter1]))
    print("vop1:\n", sess.run([vop1, fliter1]))
    print("op6:\n", sess.run([op6, fliter1]))
    print("vop6:\n", sess.run([vop6, fliter1]))
    