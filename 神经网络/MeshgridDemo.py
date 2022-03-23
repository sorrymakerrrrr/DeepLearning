# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 13:19:13 2022

@author: Xinnze
"""
import numpy as np
import matplotlib.pyplot as plt

nb_of_xs = 10
xs1 = np.linspace(-1, 8, num=nb_of_xs)
xs2 = np.linspace(-1, 8, num=nb_of_xs)
xx, yy = np.meshgrid(xs1, xs2)  # 创建网格

plt.plot(xx, yy, color='r', marker='.', linestyle='')
