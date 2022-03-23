# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 15:11:51 2022

@author: Xinnze
"""

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.reset_default_graph()

# =============================================================================
# var1 = tf.Variable(1.0, name="firstvar")
# print("var1:", var1.name)
# 
# var1 = tf.Variable(2.0, name="firstvar")
# print("var1:", var1.name)
# 
# var2 = tf.Variable(3.0)
# print("var1:", var2.name)
# 
# var2 = tf.Variable(4.0)
# print("var1:", var2.name)
# 
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print("var1 =", var1.eval())
#     print("var2 =", var2.eval())
#     
# # 上述代码定义了两次var1，可以看到内存中生成了两个var1（因为它们的name不一样），对于图来讲后面的var1是生效的
# # var2表明了：Variable定义时没有指定名字，系统会自动给加上一个名字Variable:0
# =============================================================================



# =============================================================================
# get_var1 = tf.get_variable("firstvar", [1], initializer=tf.constant_initializer(0.3))
# # tf.get_variable(name, shape, initializer)
# print("get_var1:", get_var1.name)
# 
# get_var1 = tf.get_variable("firstvar1", [1], initializer=tf.constant_initializer(0.4))
# print("get_var1:", get_var1.name)
# 
# with tf.Session() as sess2:
#     sess2.run(tf.global_variables_initializer())
#     print("get_var1 =", get_var1.eval())
# get_variable创建两个同样名字的变量是行不通的
# =============================================================================

# =============================================================================
# # get_variable 配合 variable_scope
# with tf.variable_scope("test1"):
#     var1 = tf.get_variable("firstvar", [2], dtype=tf.float32)
# 
# with tf.variable_scope("test2"):
#     var2 = tf.get_variable("firstvar", [1], dtype=tf.float32)
# 
# print("var1:", var1.name)
# print("var2:", var2.name)
# =============================================================================

# tf.reset_default_graph()
with tf.variable_scope("test1"):
    var1 = tf.get_variable("firstvar", [2], dtype=tf.float32)

    with tf.variable_scope("test2"):
        var2 = tf.get_variable("firstvar", [1], dtype=tf.float32)

print("var1:", var1.name)
print("var2:", var2.name)

# 实现共享变量功能 variable_scope 中有reuse=True属性，表示使用已经定义过的变量
# 在实际应用中，可以把var1与var2放到一个网络模型去训练，把var3与var4放到另一个网络模型去训练，
# 而两个模型的训练结果都会作用于一个模型的学习参数上
with tf.variable_scope("test1", reuse=True):
    var3 = tf.get_variable("firstvar", [2], dtype=tf.float32)

    with tf.variable_scope("test2", reuse=True):
        var4 = tf.get_variable("firstvar", [1], dtype=tf.float32)

print("var3:", var3.name)
print("var4:", var4.name)
