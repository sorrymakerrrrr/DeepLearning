# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 16:13:46 2022

@author: Xinnze
"""
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# 4.4.1 建立图
c = tf.constant(0.0)

g = tf.Graph()

with g.as_default():  # 使用tf.Graph函数来创建一个图，并在其上面定义OP
    c1 = tf.constant(0.0)
    print(c1.graph)  # 使用tf.Graph函数建立了一个新的图 并且在新建的图里添加变量 变量.graph获得所在的图
    print(g)  # 使用tf.Graph函数建立了一个新的图
    print(c.graph)  # c是在刚开始的默认图中建立的

g2 = tf.get_default_graph()  # 获得原始的默认图
print(g2)

tf.reset_default_graph()  # 新建了一张图来代替原来的默认图
g3 = tf.get_default_graph()
print(g3)

# 4.4.2 获得张量
print(c1.name)
t = g.get_tensor_by_name(name="Const:0")  # 获得图里面的张量
print(t)  # 通过对t的打印可以看到所得的t就是前面定义的张量c1
print(c1)

# 4.4.3 获取节点操作
a = tf.constant([[1.0, 2.0]])
b = tf.constant([[1.0], [3.0]])

tensor1 = tf.matmul(a, b, name="exampleop")  #tensor1是张量 不是OP

test = g3.get_tensor_by_name("exampleop:0")
print(test)

print(tensor1.op.name)
testop = g3.get_operation_by_name("exampleop")
print(testop)

with tf.Session() as sess:
    test = sess.run(test)
    print(test)
    # 先获取图 再获取当前元素
    test = tf.get_default_graph().get_tensor_by_name("exampleop:0")
    print(test)
    
# 4.4.4 获取元素列表
tt2 = g3.get_operations()  # 看图中的全部元素 get_operations来实现
print(tt2)

# 4.4.5 获取对象
# tf.Graph.as_graph_element(obj, allow_tensor=True, allow_operation=True)
# 传入一个 对象  返回一个张量或者一个OP 该函数具有验证和转换功能
tt3 = g.as_graph_element(c1)
print(tt3)

# 4.4.6 tf.get_default_graph()放在g4.as_default()的作用域里，得到的是tf.Graph()函数新建的图 而不是默认图
g4 = tf.Graph()

with g4.as_default():
    print(g)
    print(tf.get_default_graph())  

print(tf.get_default_graph())  
