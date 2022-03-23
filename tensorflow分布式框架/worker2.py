# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:47:47 2022

@author: Xinnze
"""

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

# 定义IP地址与端口 创建server
strps_hosts = "localhost:2222"  # localhost即为本机域名的写法，等同于本机IP127.0.0.1.如果是跨机器来做分布式训练，直接写成对应机器的IP地址即可
strworker_hosts = "localhost:2223, localhost:2224"

# 定义角色名称
strjob_name = "worker"
task_index = 1

# 将字符串转换为数组
ps_hosts = strps_hosts.split(',')
worker_hosts = strworker_hosts.split(',')
cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

# 创建server
server = tf.train.Server({'ps': ps_hosts, 'worker': worker_hosts},
                                    job_name=strjob_name,
                                    task_index=task_index)


# ps角色使用server.join()函数进行线程挂起，开始接受连接消息
# ps角色使用join进行等待
if strjob_name == 'ps':
    print("wait")
    server.join()


# 在创建网络结构时，tf.device函数将全部的节点都放在当前任务下
# 在tf.device函数中的任务是通过tf.train.replica_device_setter来指定的
# 在tf.train.replica_device_setter中使用worker_device来定义具体任务名称；
# 使用cluster的配置来指定角色以及对应的IP地址，从而实现整个任务下的图节点
with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d/device:CPU:0" % task_index,
        ps_device="/job:ps",
        cluster=cluster_spec)):
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    
    W = tf.Variable(tf.random_normal([1]), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")
    
    global_step = tf.train.get_or_create_global_step()  # 获得迭代次数
    
    z = tf.multiply(W, X) + b
    tf.summary.histogram('z', z)  # 将预测值以直方图显示

    cost = tf.reduce_mean(tf.square(Y - z))
    tf.summary.scalar('loss_function', cost)  # 将预测值以标量显示
    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    saver = tf.train.Saver(max_to_keep=1)
    merged_summary_op = tf.summary.merge_all()  # 合并所有summary    
    
    init = tf.global_variables_initializer()

# 创建Supervisor,管理session

# 定义参数
training_epochs = 2200
display_step = 2
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
plotdata = {"batchsize": [], "loss": []}

sv = tf.train.Supervisor(is_chief=(task_index==0),  # 0号worker为chief
                         logdir="super\\",  # 检查点文件与summary文件保存的路径
                         init_op=init,  # 使用初始化变量的函数
                         summary_op=None,  # 自动保存summary文件，这里不自动保存
                         saver=saver,  # 将保存检查点的saver对象传入，supervisor就会自动保存检查点文件。如果不想保存，设为None
                         global_step=global_step,  # 获得当前迭代的次数
                         save_model_secs=5  # 保存检查点文件的时间间隔 5s
                         )

# 连接角色目标创建session
with sv.managed_session(server.target) as sess:
    sess.run(init)
    
    print("sess ok")
    print(global_step.eval(session=sess))
    
    for epoch in range(global_step.eval(session=sess), training_epochs*len(train_X)):
        for (x, y) in zip(train_X, train_Y):
            _, epoch = sess.run([optimizer, global_step], feed_dict={X: x, Y: y})
            
            # 生成summary
            summary_str = sess.run(merged_summary_op, feed_dict={X: x, Y: y})
            
            # # 将summary写入文件
            # sv.summary_computed(sess, summary_str, global_step=epoch)
            
            if epoch % display_step == 0:
                loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
                print('Epoch: ', epoch+1, "cost= ", loss, "W= ", sess.run(W), "b= ", sess.run(b))
                
                if not loss == "NA":
                    plotdata["batchsize"].append(epoch)
                    plotdata["loss"].append(loss)
    
    print("Finished!")
    # sv.saver.save(sess, "E:\\机器学习_myself\\深度学习\\tensorflow分布式框架\\mnist_with_summarys\\" + "sv.cpk"
    #               , global_step=epoch)

sv.stop()