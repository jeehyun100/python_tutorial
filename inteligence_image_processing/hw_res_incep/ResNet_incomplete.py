from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.datasets.cifar10 import load_data
import tensorflow as tf
import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# (X_train, y_train), (X_test, y_test) = load_data()

# weight_init = tf.layers.variance_scaling_initializer()
# weight_regularizer = tf.layers.l2_regularizer(0.0001)


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

def resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='resblock') :
    with tf.variable_scope(scope) :

        x = batch_norm(x_init, is_training, scope='batch_norm_0')
        x = relu(x)

        #def conv2d(input, filters, kernel_size, strides=1, activation=tf.nn.relu, padding='SAME', name='conv'):

        if downsample :
            x = conv2d(x, channels, kernel=3, stride=2, name='conv_0')
            x_init = conv2d(x_init, channels, kernel=1, stride=2, name='conv_0')

        else :
            x = conv2d(x, channels, kernel=3, stride=1, name='conv_0')

        x = batch_norm(x, is_training, scope='batch_norm_1')
        x = relu(x)
        x = conv2d(x, channels, kernel=3, stride=1, name='conv_1')



        return x + x_init

def res_block(x, filter_num, kernel_size, strides, name='res_block', is_training=True, downsample=False, scope='resblock' ):
    ### Assignment ###
    ### Use conv2d function and residual variable
    #return out stride1, kernel_size = 3
    with tf.variable_scope(scope) :

        res_x = batch_norm(x, is_training, scope='batch_norm_0')
        res_x = relu(res_x)

        if downsample :
            res_x = conv2d(res_x, filter_num, kernel_size, strides=strides, name='conv_0')
            x = conv2d(x, filter_num, kernel=1, stride=2, name='conv_init')

        else :
            res_x = conv2d(res_x, filter_num, kernel_size, strides=strides, name='conv_0')

        res_x = batch_norm(res_x, is_training, scope='batch_norm_1')
        res_x = relu(res_x)
        res_x = conv2d(res_x, filter_num, kernel_size, strides=strides, name='conv_1')

        return res_x + x

#
# def fully_conneted(x, units, use_bias=True, scope='fully_0'):
#     with tf.variable_scope(scope):
#         x = flatten(x)
#         x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)
#
#         return x

def get_residual_layer(res_n) :
    x = []

    if res_n == 18 :
        x = [2, 2, 2, 2]

    if res_n == 34 :
        x = [3, 4, 6, 3]

    if res_n == 50 :
        x = [3, 4, 6, 3]

    if res_n == 101 :
        x = [3, 4, 23, 3]

    if res_n == 152 :
        x = [3, 8, 36, 3]

    return x

##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=True, scope='batch_norm'):
    # return tf_contrib.layers.batch_norm(x,
    #                                     decay=0.9, epsilon=1e-05,
    #                                     center=True, scale=True, updates_collections=None,
    #                                     is_training=is_training, scope=scope)
    # return tf.nn.batch_normalization(x,
    # mean,
    # variance,
    # offset,
    # scale,
    # variance_epsilon,
    # name=None
    #def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    #with tf.variable_scope('bn'):
        # beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
        #                    name='beta', trainable=True)
        # gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
        #                     name='gamma', trainable=True)
        # batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        # ema = tf.train.ExponentialMovingAverage(decay=0.5)
        #
        # def mean_var_with_update():
        #     ema_apply_op = ema.apply([batch_mean, batch_var])
        #     with tf.control_dependencies([ema_apply_op]):
        #         return tf.identity(batch_mean), tf.identity(batch_var)
        #
        # mean, var = tf.cond(phase_train,
        #                     mean_var_with_update,
        #                     lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.layers.batch_normalization(x,
                                        training=is_training,
                                        name = scope)
    return normed

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

def relu(x):
    return tf.nn.relu(x)


def res_net(x):
    ### Assignment ###
    ### Use conv2d, res_block, max_pool, dense and other function

    with tf.variable_scope("network"):

        # if self.res_n < 50:
        #     residual_block = resblock
        # else:
        #     residual_block = bottle_resblock
        res_n = 18
        residual_list = get_residual_layer(res_n)

        filter_num = 32  # paper is 64
        kernel_size = 3
        strides = 1
        label_dim = 10

        #conv2d(res_x, filter_num, kernel_size, strides=strides, name='conv_0')
        x = conv2d(x, filter_num, 3, strides=1, name='conv')

        #res_block(x, filter_num, kernel_size, strides, name='res_block', is_training=True, downsample=False, scope='resblock' ):
    ### Assignment ###

        for i in range(residual_list[0]):
            x = res_block(x, filter_num = filter_num, kernel_size = kernel_size, strides = strides, is_training=True, scope='resblock0_' + str(i))

        x = batch_norm(x, is_training = True, scope='batch_norm')
        x = relu(x)

        x = global_avg_pooling(x)
        x = dense(x, label_dim, name='logit')

    return x


##################################################################################
# Layer
##################################################################################
#
#
# def conv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
#     with tf.variable_scope(scope):
#         x = tf.layers.conv2d(inputs=x, filters=channels,
#                              kernel_size=kernel, kernel_initializer=weight_init,
#                              kernel_regularizer=weight_regularizer,
#                              strides=stride, use_bias=use_bias, padding=padding)
#
#         return x
#
# def fully_conneted(x, units, use_bias=True, scope='fully_0'):
#     with tf.variable_scope(scope):
#         x = flatten(x)
#         x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)
#
#         return x
#
# def resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='resblock') :
#     with tf.variable_scope(scope) :
#
#         x = batch_norm(x_init, is_training, scope='batch_norm_0')
#         x = relu(x)
#
#
#         if downsample :
#             x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
#             x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')
#
#         else :
#             x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')
#
#         x = batch_norm(x, is_training, scope='batch_norm_1')
#         x = relu(x)
#         x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')
#
#         return x + x_init
#
#
# def bottle_resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='bottle_resblock') :
#     with tf.variable_scope(scope) :
#         x = batch_norm(x_init, is_training, scope='batch_norm_1x1_front')
#         shortcut = relu(x)
#
#         x = conv(shortcut, channels, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_front')
#         x = batch_norm(x, is_training, scope='batch_norm_3x3')
#         x = relu(x)
#
#         if downsample :
#             x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
#             shortcut = conv(shortcut, channels*4, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')
#
#         else :
#             x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')
#             shortcut = conv(shortcut, channels * 4, kernel=1, stride=1, use_bias=use_bias, scope='conv_init')
#
#         x = batch_norm(x, is_training, scope='batch_norm_1x1_back')
#         x = relu(x)
#         x = conv(x, channels*4, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_back')
#
#         return x + shortcut



def get_residual_layer(res_n) :
    x = []

    if res_n == 18 :
        x = [2, 2, 2, 2]

    if res_n == 34 :
        x = [3, 4, 6, 3]

    if res_n == 50 :
        x = [3, 4, 6, 3]

    if res_n == 101 :
        x = [3, 4, 23, 3]

    if res_n == 152 :
        x = [3, 8, 36, 3]

    return x



# ##################################################################################
# # Sampling
# ##################################################################################
#
# def flatten(x) :
#     return tf.layers.flatten(x)
#
#
#
# def avg_pooling(x) :
#     return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')
#
# ##################################################################################
# # Activation function
# ##################################################################################
#
#
# def relu(x):
#     return tf.nn.relu(x)




epochs = 100
learning_rate = 0.00001
batch_size = 64
dropout = 0.8


# tf Graph input
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.int32, [None, 10])
# y_one_hot = tf.squeeze(tf.one_hot(y, 10),axis=1)
#phase_train = tf.placeholder(tf.bool, name='phase_train')


keep_prob = tf.placeholder(tf.float32, [])  # dropout (keep probability)
x_resize = tf.image.resize_images(x, [64, 64])

# Construct model
pred = res_net(x)

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
        #64.28.28.1
        #64,10
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
    saver.save(sess, './resnet')