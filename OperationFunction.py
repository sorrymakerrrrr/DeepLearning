# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:20:45 2022

@author: Xinnze
"""
import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

# 类型变换相关函数
W = tf.Variable(1.0)
a = tf.cast(W, tf.int32, name='ToInt32')
b = tf.to_double(W, name='ToDouble')
print(W, '\n', a, '\n', b)

# 形状变换相关函数
t = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(np.shape(t))
tshape = tf.shape(t)
print(tshape)  # 返回一个张量 值为python自有类型t的 shape
tshape2 = tf.shape(tshape)
print(tshape2) # 返回一个张量 值为张量Tshape的 shape
with tf.Session() as sess:
    print(sess.run(tshape))  # 输出[9] 表示tshape的 值
    print(sess.run(tshape2))  # 输出[1] 表示tshape2的 值


t1 = [[[1,1,1,1], [2,2,2,2], [3,3,3,3]], [[4,4,4,4], [5,5,5,5], [6,6,6,6]]]
a = [[2,3,3], [1,5,5]]
b = [[[[2], [1]]]]
with tf.Session() as sess2:
    print(sess2.run(tf.shape(t1)))
    
    sizet = tf.size(t1)
    print(sess2.run(sizet))  # 输出24， 表示列表t中元素的 个数 
    
    rankt = tf.rank(t1)
    print(sess2.run(rankt))  # 输出3， 表示列表t的 阶  (一共3层括号表示3阶)
    
    tt = tf.reshape(t, [3, 3])  # 将原有数据的shape按照指定形状进行 变化 ，生成一个新的张量 相乘必须为张量元素的总个数
    print(sess2.run(tt))
    ttt = tf.reshape(t, [1, -1])
    print(sess2.run(ttt))  # -1表示在该维度下按照原有数据 自动计算
    
    # 插入维度1进入tensor中
    a0 = tf.expand_dims(a, 0)  # 零个括号之后加括号 shape:(1,2,3)
    a1 = tf.expand_dims(a, 1)  # 一个括号之后加括号 shape:(2,1,3)
    a2 = tf.expand_dims(a, 2)  # 二个括号之后加括号 shape:(2,3,1)
    print(sess2.run(a0), '\n', sess2.run(a1), '\n',sess2.run(a2))
    
    # 将dim指定的 维度去掉  其中dim指定的维度 必须为1 ，否则报错
    b0 = tf.squeeze(b, 0)
    b1 = tf.squeeze(b, 1)
    # b1 = tf.squeeze(b, 2)  # 报错
    b3 = tf.squeeze(b, 3)
    b4 = tf.squeeze(b, -1)
    print(np.shape(b0), '\n', np.shape(b1), '\n', np.shape(b3), '\n', np.shape(b4))



# 数据操作相关函数

c = [[[1,1,1], [2,2,2]], [[3,3,3],[4,4,4]], [[5,5,5], [6,6,6]]]
d1 = [[1,2,3], [4,5,6]]
d2 = [[7,8,9], [10,11,12]]
y = tf.constant([0., 2., -1.])
with tf.Session() as sess3:
    # 对input进行 切片 操作 tf.slice(input, begin, size)
    slicec1 = tf.slice(c, [1,0,0], [1,1,3])
    slicec2 = tf.slice(c, [1,0,0], [1,2,3])
    slicec3 = tf.slice(c, [1,0,0], [2,1,3])
    print(sess3.run(slicec1), '\n')
    print(sess3.run(slicec2), '\n')
    print(sess3.run(slicec3), '\n')
    
    # tf.split(value, num_or_size_splits, axis=0, num=None, name=split)
    # 沿着某一维度将tensor 分离 成列表长度个数的张量，每个张量在分割维度的元素为列表内的每个值
    split0, split1, split2 = tf.split(c, [1,1,1], 0)
    print(sess3.run(split0), '\n')
    print(sess3.run(split1), '\n')
    print(sess3.run(split2), '\n')
    
    # 沿着某一维度 连接 tensor 
    d3 = tf.concat([d1, d2], 0)  # 沿第0维度连接，原：d1:2*3 d2:2*3 现：d3:4*3
    print(sess3.run(d3), '\n')
    
    # tf.stack(input, axis=0) 将两个维张量沿着axis轴 组合 成一个N+1维张量
    d4 = tf.stack([d1, d2], axis=0)  # 原(2,3) 现(2,2,3)  插第一个2
    d5 = tf.stack([d1, d2], axis=1)  # 原(2,3) 现(2,2,3)  插第二个2
    d6 = tf.stack([d1, d2], axis=2)  # 原(2,3) 现(2,3,2)
    print(sess3.run(d4), '\n')
    print(sess3.run(d5), '\n')
    print(sess3.run(d6), '\n')
    
    # tf.unstack(value, num=None, axis=0)  按指定的维数进行 拆分  num表示输出的列表的个数，可以忽略
    d7 = tf.unstack(d1, axis=0)  # d1 2*3  按第0个维度分分成2个
    d8 = tf.unstack(d1, axis=1)  # 按第1个维度分分成3个
    
    # tf.gather(params, indices, validate_indices=None, name=None) 根据索Z引来切片
    y1 = tf.gather(y, [2, 0])
    print(sess3.run(y1))

e = np.array([[[1,2,3], [4,5,6]]])
with tf.Session() as sess4:
    print(sess4.run(tf.transpose(e, [1,0,2])))
    
