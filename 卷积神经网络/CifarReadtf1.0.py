# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 14:09:47 2022
从cifar二进制文件中读取
@author: Xinnze
"""

from cifar10 import cifar10_input
import tensorflow.compat.v1 as tf
import pylab


tf.disable_v2_behavior()
batch_size = 128 
data_dir = 'cifar-10-batches-bin'

# 会将图片剪裁好，由32*32*3变成24*24*3 之后又进行了一次图形标准化
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

# print(images_test.shape)
# =============================================================================
# 不能这么使用 因为带with语法的Session是自动关闭的，当运行结束后里面的所有操作都关掉
# 而此时的队列还在等待另一个进程往里写数据 所以会报错
# with tf.Session() as sess:
# =============================================================================
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 运行队列(启动线程)，向队列里面读取数据
tf.train.start_queue_runners()
# 从队列里拿出指定批次的数据
image_batch, label_batch = sess.run([images_test, labels_test])
print("__\n", image_batch[0])
print("__\n", label_batch[0])
pylab.imshow(image_batch[0])
pylab.show()
