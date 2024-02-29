# 引入三方库
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1、定义输入层
# 没有数据，用numpy生成200个随机数来当作数据
x_data = np.linspace(-0.5, 0.5, 200)[:,
         np.newaxis]  # 用linspace函数生成200个-0.5到0.5之间的一维数组，然后用np.newaxis增加列维（[np.newaxis，：] np.newaxis在前面就是增加行维，在后面就是增加列维），变成（200，1）的矩阵
noise = np.random.normal(0, 0.02, x_data.shape)  # np.random.normal函数生成 x_data.shape 个正态分布的随机数（此处当作噪音）
y_data = np.square(x_data) + noise  # 将生成的噪音+ x_data 的平方，当作y的值
# x，y用placeholder定义占位，用于当作输入的参数变量
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 2、定义隐藏层
# 定义权重矩阵w1 偏置矩阵b1
w1 = tf.Variable(tf.random_normal([1, 10]))
b1 = tf.Variable(tf.zeros([1, 10]))
# 计算wx+b
res = tf.matmul(x, w1) + b1
# 激活函数，使用tanh
L1 = tf.nn.tanh(res)

# 3、定义输出层
w2 = tf.Variable(tf.random_normal([10, 1]))
b2 = tf.Variable(tf.zeros([1, 1]))
# 计算wx+b
res2 = tf.matmul(L1, w2) + b2
# 激活函数，使用tanh
prediction = tf.nn.tanh(res2)

# 4、根据损失函数求loss(均方差函数)
loss = tf.reduce_mean(tf.square(y - prediction))
# 5、根据loss和梯度下降法反向传播更新权重（训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 最后执行
with tf.Session() as sess:
    # 变量全都初始化
    sess.run(tf.global_variables_initializer())
    # 训练2000次
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
    # 训练完成后，推理
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
    plt.show()
