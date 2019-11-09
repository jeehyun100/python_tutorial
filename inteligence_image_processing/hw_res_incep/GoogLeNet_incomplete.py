from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.datasets.cifar10 import load_data
import tensorflow as tf
import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# (X_train, y_train), (X_test, y_test) = load_data()

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
                         activation=activation, name=name)
    return out

def dense(input, unit, activation=tf.nn.relu, name='dense'):
    with tf.variable_scope(name, reuse=False):
        out = tf.layers.dense(input, unit, activation=activation, name=name)
    return out

def max_pool(input, k, s, name='pool'):
    out = tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)
    return out

#
# @add_arg_scope
# def inception_layer(conv_11_size, conv_33_reduce_size, conv_33_size,
#                     conv_55_reduce_size, conv_55_size, pool_size,
#                     layer_dict, inputs=None,
#                     bn=False, wd=0, init_w=None,
#                     pretrained_dict=None, trainable=True, is_training=True,
#                     name='inception'):
#
# def max_pool(x, name, filter_height = 3, filter_width = 3,
# 	stride = 2, padding = 'VALID'):
#
# 	"""Create a max pooling layer."""
#
# 	return tf.nn.max_pool(x, ksize = [1, filter_height, filter_width, 1],
# 		strides = [1, stride, stride, 1], padding = padding, name = name)



def inception_block(x, conv11_size, conv33_rsize, conv33_size, conv55_rsize, conv55_size, pool_size, name='incept'):
    ### Assignment ###
    ### Use conv2d and max_pool function

    with tf.variable_scope(name) as scope:
        #def conv2d(input, filters, kernel_size, strides=1, activation=tf.nn.relu, padding='SAME', name='conv'):
        conv_1 = conv2d(x, conv11_size, kernel_size=1, name='{}_1x1'.format(name))

        # conv_3_reduce = conv_layer(x, filter_height=1, filter_width=1,
        #                            num_filters=conv_3_reduce_size, name='{}_3x3_reduce'.format(name))
        conv_3_reduce = conv2d(x, conv33_rsize, kernel_size=1, name='{}_3x3_reduce'.format(name))

        conv_3 = conv2d(conv_3_reduce, conv33_size, kernel_size=3, name='{}_3x3'.format(name))

        conv_5_reduce = conv2d(x, conv55_rsize, kernel_size=1, name='{}_5x5_reduce'.format(name))

        conv_5 = conv2d(conv_5_reduce, conv55_size, kernel_size = 5, name='{}_5x5'.format(name))

        pool = max_pool(x, 3, 1, name='{}_pool'.format(name))
        #pool = max_pool(x, stride =1, padding = 'SAME', name = '{}_pool'.format(name))

        pool_proj = conv2d(pool, pool_size, kernel_size=1, name='{}_pool_proj'.format(name))

        out = tf.concat([conv_1, conv_3, conv_5, pool_proj], axis=3, name='{}_concat'.format(name))

    return out

def googlenet(x):
    ### Assignment ###
    ### Use cpmv2d, max_pool, inception_block, dens and dropout function
    # 3, filter_width = 3,stride = 2
    conv1 = conv2d(x, 64, kernel_size=5, strides=1, name='conv1')
    pool1 = max_pool(conv1, 3, 2, name='pool1')

    inception2a = inception_block(pool1, 64, 96, 128, 16, 32, 32, name='inception1a')
    inception2b = inception_block(inception2a, 128, 128, 192, 32, 96, 64, name='inception1b')
    pool2 = max_pool(inception2b,3, 2, name='pool2')
    inception3a = inception_block(pool2, 192, 96, 208, 16, 48, 64, name='inception3a')
    inception3b = inception_block(inception3a, 160, 112, 224, 24, 64, 64, name='inception3b')
    gap = tf.nn.avg_pool(inception3b, ksize=[1, 6, 6, 1], strides=[1, 1, 1, 1], padding='VALID', name='gap')
    gap_dropout = tf.nn.dropout(gap, keep_prob=keep_prob)
    flatten = tf.layers.flatten(gap_dropout)
    dense2 = dense(flatten, 10, name='fc4')

    return dense2



epochs = 100
learning_rate = 0.00001
batch_size = 64
dropout = 0.8


# tf Graph input
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.int32, [None, 10])
#hold_prob = tf.placeholder(tf.float32)
# y_one_hot = tf.squeeze(tf.one_hot(y, 10), axis=1)

keep_prob = tf.placeholder(tf.float32, [])  # dropout (keep probability)
x_resize = tf.image.resize_images(x, [64, 64])

# Construct model
pred = googlenet(x)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
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

    print("Optimization Finished!")
    saver.save(sess, './goolgenet')