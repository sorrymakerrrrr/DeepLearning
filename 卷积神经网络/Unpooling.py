# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 19:10:58 2022
反池化操作
tensorflow1.x中没有反池化操作的函数  但是同样有一个池化的反向传播函数tf.nn.max_pool_with_argmax可以输出位置
开发者可以利用此函数做一些改动 封装一个最大池化操作，再根据mask写出反池化函数
@author: Xinnze
"""

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def max_pool_with_argmax(net, stride):
    # 调用tf.nn.max_pool_with_argmax函数获得每个最大值的位置mask
    _, mask = tf.nn.max_pool_with_argmax(net, ksize=[1, stride, stride, 1],
                                         strides=[1, stride, stride, 1], padding='SAME')

    # 将反向传播的mask梯度停止
    mask = tf.stop_gradient(mask)

    # 调用tf.nn.max_pool函数计算最大池化操作
    net = tf.nn.max_pool(net, ksize=[1, stride, stride, 1],
                         strides=[1, stride, stride, 1], padding='SAME')

    # 返回mask和池化结果  mask的值是将整个数组flat(扁平化)之后的索引 但却保持与池化结果一致的shape
    return net, mask


# 定义数组看自己封装的函数输出什么
img = tf.constant([
    [[0.0, 4.0], [0.0, 4.0], [0.0, 4.0], [0.0, 4.0]],
    [[1.0, 5.0], [1.0, 5.0], [1.0, 5.0], [1.0, 5.0]],
    [[2.0, 6.0], [2.0, 6.0], [2.0, 6.0], [2.0, 6.0]],
    [[3.0, 7.0], [3.0, 7.0], [3.0, 7.0], [3.0, 7.0]],
])
img = tf.reshape(img, [1, 4, 4, 2])

pooling2 = tf.nn.max_pool(img, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
encode, mask = max_pool_with_argmax(img, 2)

with tf.Session() as sess:
    # print("image: ")
    image = sess.run(img)
    # print(image)

    result = sess.run(pooling2)
    print("pooling2: \n", result)

    result2, mask2 = sess.run([encode, mask])
    print("encode: \n", result2, '\n', mask2)


# 定义反池化操作
def unpool(net, mask, stride):  # mask.shape = net.shape = [batch, height, width, channels]
    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()

    # 计算new shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1],
                    input_shape[2] * ksize[2], input_shape[3])

    # 计算索引
    # tf.ones_like(tensor, dtype=None, name=None, optimize=True): 创建一个和tensor维度一样 元素都为1的张量
    one_like_mask = tf.ones_like(mask)  # one_like_mask.shape = [batch, height, width, channels]

    # tf.range(a)与range(a)一样
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range  # 第0维的索引
    # print(b, '\n')
    
    y = mask // (output_shape[2] * output_shape[3])  # 第一维的索引
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]  # 第二维的索引
    # print(x, '\n')
    # print(y, '\n')

    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range  # 第三维的索引
    # print(f, '\n')

    # 转置索引
    # tf.size(net) 计算net的元素数
    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
# =============================================================================
#     tf.scatter_nd(indices, values, output_shape)  根据indices将values散布到新的（初始为零）张量。
#     indices: 插入元素索引 
#     values: 插入的元素  索引定位到第几个维度，插入元素的个数 就是 那个维度的元素的个数
#     output_shape: 插入的张量的shape
# =============================================================================
    ret = tf.scatter_nd(indices, values, output_shape)
    # with tf.Session() as sess:
    #     print(sess.run(indices), '\n')
    #     print(sess.run(values), '\n')
    return ret


img2 = unpool(encode, mask2, 2)
with tf.Session() as sess:
    print('result: \n', sess.run(img2))
