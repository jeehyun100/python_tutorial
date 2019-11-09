from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.datasets.cifar10 import load_data
import tensorflow as tf
import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

mnist = input_data.read_data_sets("MNIST_data/", reshape=False, one_hot=True)
X_train, y_train = mnist.train.images, mnist.train.labels
X_valid, y_valid = mnist.validation.images, mnist.validation.labels
X_test, y_test = mnist.test.images, mnist.test.labels

print("Image Shape: {}".format(X_train[0].shape))
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_valid)))
print("Test Set:       {} samples".format(len(X_test)))


def conv2d(input, filters, kernel_size, strides=1, activation=tf.nn.relu, padding='SAME', name='conv'):
    with tf.variable_scope(name, reuse=False):
        out = tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                         activation=activation)
    return out


def dense(input, unit, activation=tf.nn.relu, name='dense'):
    with tf.variable_scope(name, reuse=False):
        out = tf.layers.dense(input, unit, activation=activation, name=name)
    return out


def max_pool(input, k, s, name='pool'):
    out = tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)
    return out


def res_block(x, filter_num, kernel_size, strides, name='res_block', is_training=True, downsample=False, scope='resblock' ):

    with tf.variable_scope(scope):

        res_x = batch_norm(x, is_training, scope='batch_norm_0')
        res_x = relu(res_x)

        if downsample :
            res_x = conv2d(res_x, filter_num, kernel_size=3, strides=2, name='conv_0')
            x = conv2d(x, filter_num, kernel_size=1, strides=2, name='conv_init')

        else :
            res_x = conv2d(res_x, filter_num, kernel_size, strides=strides, name='conv_0')

        res_x = batch_norm(res_x, is_training, scope='batch_norm_1')
        res_x = relu(res_x)
        res_x = conv2d(res_x, filter_num, kernel_size, strides=strides, name='conv_1')

        return res_x + x


def get_residual_layer(res_n) :
    x = []

    if res_n == 18 :
        x = [2, 2, 2, 2]

    return x


def batch_norm(x, is_training=True, scope='batch_norm'):
    with tf.variable_scope(scope):
        normed = tf.layers.batch_normalization(x,
                                            training=is_training)
    return normed


def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap


def relu(x):
    return tf.nn.relu(x)


def res_net(x, is_training=True, reuse = False):
    ### Assignment ###
    ### Use conv2d, res_block, max_pool, dense and other function

    with tf.variable_scope("network", reuse=reuse):
        res_n = 18
        residual_list = get_residual_layer(res_n)

        filter_num = 32  # paper is 64
        kernel_size = 3
        strides = 1
        label_dim = 10

        x = conv2d(x, 32, 3, strides=1, name='conv')

        for i in range(residual_list[0]):
            x = res_block(x, filter_num = filter_num, kernel_size = kernel_size, strides = strides, is_training=is_training, scope='resblock0_' + str(i))

        x = res_block(x, filter_num=filter_num * 2, kernel_size = kernel_size, strides = strides, is_training=is_training, downsample=True, scope='resblock1_0')

        for i in range(1, residual_list[1]):
            x = res_block(x, filter_num=filter_num * 2, kernel_size = kernel_size, strides = strides, is_training=is_training, downsample=False,
                               scope='resblock1_' + str(i))


        x = res_block(x, filter_num=filter_num * 4, kernel_size = kernel_size, strides = strides, is_training=is_training, downsample=True, scope='resblock2_0')

        for i in range(1, residual_list[2]):
            x = res_block(x, filter_num=filter_num * 4, kernel_size = kernel_size, strides = strides, is_training=is_training, downsample=False,
                               scope='resblock2_' + str(i))


        x = res_block(x, filter_num=filter_num * 8, kernel_size = kernel_size, strides = strides, is_training=is_training, downsample=True, scope='resblock_3_0')

        for i in range(1, residual_list[3]):
            x = res_block(x, filter_num=filter_num * 8, kernel_size = kernel_size, strides = strides, is_training=is_training, downsample=False,
                               scope='resblock_3_' + str(i))

        x = batch_norm(x, is_training=is_training, scope='batch_norm')

        x = global_avg_pooling(x)
        x = tf.nn.dropout(x, keep_prob=keep_prob)
        x = tf.layers.flatten(x)
        x = dense(x, label_dim, name='logit')

    return x


def get_residual_layer(res_n):
    x = []
    if res_n == 18:
        x = [2, 2, 2, 2]
    return x


epochs = 10
learning_rate = 0.001
batch_size = 500
dropout = 0.8


# tf Graph input
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.int32, [None, 10])

keep_prob = tf.placeholder(tf.float32, [])  # dropout (keep probability)
x_resize = tf.image.resize_images(x, [64, 64])

# Construct model
pred = res_net(x)
test_pred = res_net(x, is_training=False, reuse=True)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, -1), tf.argmax(y, -1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

saver = tf.train.Saver()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                      gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))) as sess:
    sess.run(init)
    # Keep training until reach max iterations
    print('Training...')
    for i in range(epochs):
        s = np.random.permutation(len(X_train))
        X_train = X_train[s]
        y_train = y_train[s]
        loss_total, acc_total = 0, 0
        for offset in range(0, len(X_train), batch_size):
            end = offset + batch_size
            batch_xs, batch_ys = X_train[offset:end], y_train[offset:end]

            # Fit training using batch data
            _, acc, loss = sess.run([optimizer, accuracy, cost], feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})

            # Calculate batch accuracy
            # Calculate batch loss
            acc_total += acc
            loss_total += loss

        acc_total /= (len(X_train) / batch_size)
        loss_total /= (len(X_train) / batch_size)
        print("epoch: " + str(i) + ", Training Loss= " + "{:.4f}".format(
            loss_total) + ", Training Accuracy= " + "{:.4f}".format(acc_total))

    test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images[: 1000],
                                             y: mnist.test.labels[: 1000], keep_prob: 1.0})

    print("test_Accuracy : {0}".format(test_accuracy))
    print("Optimization Finished!")
    saver.save(sess, './resnet')
