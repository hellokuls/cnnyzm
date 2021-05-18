import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from process import next_batch
from getimg import CAPTCHA_HEIGHT, CAPTCHA_WIDTH, CAPTCHA_LEN, CAPTCHA_LIST
from datetime import datetime


# 随机生成权重
def weight_variable(shape, w_alpha=0.01):
    initial = w_alpha * tf.random_normal(shape)
    return tf.Variable(initial)


# 随机生成偏置项
def bias_variable(shape, b_alpha=0.1):
    initial = b_alpha * tf.random_normal(shape)
    return tf.Variable(initial)


# 局部变量线性组合，步长为1，模式‘SAME’代表卷积后图片尺寸不变，即零边距

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# max pooling,取出区域内最大值为代表特征， 2x2pool，图片尺寸变为1/2

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 三层卷积神经网络计算图
def cnn_graph(x, keep_prob, size, captcha_list=CAPTCHA_LIST, captcha_len=CAPTCHA_LEN):
    # 图片reshape为4维向量
    image_height, image_width = size
    x_image = tf.reshape(x, shape=[-1, image_height, image_width, 1])
    # 第一层

    # filter定义为3x3x1， 输出32个特征, 即32个filter

    w_conv1 = weight_variable([3, 3, 1, 32])

    b_conv1 = bias_variable([32])

    # rulu激活函数

    h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_image, w_conv1), b_conv1))

    # 池化

    h_pool1 = max_pool_2x2(h_conv1)
    # dropout防止过拟合

    h_drop1 = tf.nn.dropout(h_pool1, keep_prob)

    # 第二层
    w_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_drop1, w_conv2), b_conv2))
    h_pool2 = max_pool_2x2(h_conv2)

    h_drop2 = tf.nn.dropout(h_pool2, keep_prob)

    # 第三层

    w_conv3 = weight_variable([3, 3, 64, 64])

    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(tf.nn.bias_add(conv2d(h_drop2, w_conv3), b_conv3))
    h_pool3 = max_pool_2x2(h_conv3)
    h_drop3 = tf.nn.dropout(h_pool3, keep_prob)

    # 全连接层

    image_height = int(h_drop3.shape[1])
    image_width = int(h_drop3.shape[2])
    w_fc = weight_variable([image_height * image_width * 64, 1024])
    b_fc = bias_variable([1024])
    h_drop3_re = tf.reshape(h_drop3, [-1, image_height * image_width * 64])
    h_fc = tf.nn.relu(tf.add(tf.matmul(h_drop3_re, w_fc), b_fc))
    h_drop_fc = tf.nn.dropout(h_fc, keep_prob)


    # 全连接层(输出层)

    w_out = weight_variable([1024, len(captcha_list) * captcha_len])
    b_out = bias_variable([len(captcha_list) * captcha_len])
    y_conv = tf.add(tf.matmul(h_drop_fc, w_out), b_out)

    return y_conv


# 最小化loss

def optimize_graph(y, y_conv):
    # 交叉熵计算loss

    # sigmod_cross适用于每个类别相互独立但不互斥，如图中可以有字母和数字
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv, labels=y))
    # 最小化loss优化
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    return optimizer


# 偏差计算

def accuracy_graph(y, y_conv, width=len(CAPTCHA_LIST), height=CAPTCHA_LEN):
    # 预测值

    predict = tf.reshape(y_conv, [-1, height, width])

    max_predict_idx = tf.argmax(predict, 2)
    # 标签

    label = tf.reshape(y, [-1, height, width])

    max_label_idx = tf.argmax(label, 2)

    correct_p = tf.equal(max_predict_idx, max_label_idx)

    accuracy = tf.reduce_mean(tf.cast(correct_p, tf.float32))

    return accuracy


# 训练cnn

def train(height=CAPTCHA_HEIGHT, width=CAPTCHA_WIDTH, y_size=len(CAPTCHA_LIST) * CAPTCHA_LEN):
    acc_rate = 0.95

    # 按照图片大小申请占位符

    x = tf.placeholder(tf.float32, [None, height * width])
    y = tf.placeholder(tf.float32, [None, y_size])

    # 防止过拟合 训练时启用 测试时不启用
    keep_prob = tf.placeholder(tf.float32)

    # cnn模型

    y_conv = cnn_graph(x, keep_prob, (height, width))

    # 最优化

    optimizer = optimize_graph(y, y_conv)

    # 偏差

    accuracy = accuracy_graph(y, y_conv)

    # 启动会话.开始训练

    saver = tf.train.Saver()

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    step = 0
    while 1:
        # 每批次64个样本

        batch_x, batch_y = next_batch(64)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})
        print("step：", step)
        # 每训练一百次测试一次
        if step % 100 == 0:
            batch_x_test, batch_y_test = next_batch(100)
            acc = sess.run(accuracy, feed_dict={x: batch_x_test, y: batch_y_test, keep_prob: 1.0})
            print(datetime.now().strftime('%c'), ' step:', step, ' accuracy:', acc)
            # 偏差满足要求，保存模型
            if acc > acc_rate:
                model_path = os.getcwd() + os.sep + str(acc_rate) + "captcha.model"
                saver.save(sess, model_path, global_step=step)
                acc_rate += 0.01
                if acc_rate > 0.99:
                    break
        step += 1
    sess.close()

if __name__ == '__main__':
    train()

