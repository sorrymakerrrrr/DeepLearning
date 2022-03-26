# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 10:03:10 2022

@author: Xinnze
"""
from tensorflow.keras import datasets
import tensorflow.compat.v1 as tf 
import pylab
import numpy as np


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images = train_images # 标准化
test_images = test_images
# =============================================================================
# 将cifar-10-batchs-py下载到这个文件夹中 并且读取数据
# 可以用from tensorflow.keras import datasets 
#      (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data() 来获取
# 但是文件会存储在c盘 C:\Users\Xinnze\.keras中
# =============================================================================
def load_data(dic):
    
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            info = pickle.load(fo, encoding='bytes')
        return info   

    for i in range(1, 6):
        info = unpickle(dic + '\\data_batch_' + str(i))
        if i == 1:
            images_train = info[b'data']
            labels_train = info[b'labels']
        else:
            images_train = np.concatenate([images_train, info[b'data']], axis=0)
            labels_train = np.concatenate([labels_train, info[b'labels']], axis=0)
    
    images_train = np.reshape(images_train, [-1, 3, 32, 32]).transpose(0, 2, 3, 1)  # reshape之后再进行transpose操作 重点以及难点
    labels_train = np.reshape(labels_train, [-1, 1])
    
    info2 = unpickle(dic + '\\test_batch')
    images_test = np.reshape(info2[b'data'], [-1, 3, 32, 32]).transpose(0, 2, 3, 1)
    labels_test = np.reshape(info2[b'labels'], [-1, 1])
    
    return (images_train, labels_train), (images_test, labels_test)
        

(images_train, labels_train), (images_test, labels_test) = load_data('cifar-10-batches-py')

# =============================================================================
# # 验证自己写的函数是否正确
# print(np.array_equal(train_labels, labels_train))
# print(np.array_equal(images_train, train_images))
# print(np.array_equal(images_test, test_images))
# print(np.array_equal(labels_test, test_labels))
# =============================================================================

