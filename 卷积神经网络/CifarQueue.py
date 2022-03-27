# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:06:30 2022
cifar队列协调器
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


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    # 定义协调器
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    
    image_batch, label_batch = sess.run([images_test, labels_test])
    print('__\n', image_batch[0])
    print('__\n', label_batch[0])
    pylab.imshow(image_batch[0])
    pylab.show()
    
    coord.request_stop()
    