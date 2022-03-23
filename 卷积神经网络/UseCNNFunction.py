# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:47:17 2022

@author: Xinnze
"""
"""
tf.nn.conv2d(input, fliter, strides, padding, use_cudnn_on_gpu=None, name=None)
input: [batch, in_height, in_width, in_channels]  type:Tensor  要求类型float32, float64
fliter: [fliter_height, fliter_width, in_channels, out_channels]  type:Tensor
         卷积核高度      卷积核宽度     图像通道数    卷积核个数
padding: "SAME"  "VALUE"

"""
