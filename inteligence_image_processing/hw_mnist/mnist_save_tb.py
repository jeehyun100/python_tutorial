import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
from PIL import Image, ImageFilter
import os
import numpy as np
import matplotlib.pyplot as plt  # for plotting
from sklearn.metrics import f1_score
import matplotlib.gridspec as gridspec


class Model(object):

    def __init__(self, sess, conf):
        """
        This function init variable
        """
        self.output_layer = None
        self.hypothesis = None
        self.generative_hypothesis = None
        self.dataset = None
        self.accuracy_op = None
        self.X = None
        self.Y = None
        self.G = None
        self.Z = None
        self.d_train_step = None
        self.g_train_step = None
        self.d_loss = None
        self.g_loss = None
        self.saver = None
        self.writer = None
        self.cost = None
        self.optimizer = None
        self.output_shape = None
        self.input_shape = None
        self.input_latent_shape = None
        self.latent_shape = None
        self.input_x_shape = None
        self.best_val_acc = 0
        self.best_gloval_step = 0

        self.sess = sess
        self.conf = conf
        self.def_params()
        self.model_dir = conf.modeldir + "/" + conf.model_name + "_" + conf.active_func + "_" + conf.initial_type + "_" + str(
            conf.learning_rate)
        self.log_dir = conf.logdir + "/" + conf.model_name + "_" + conf.active_func + "_" + conf.initial_type + "_" + str(
            conf.learning_rate)
        self.gen_dir = conf.gendir + "/"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.gen_dir):
            os.makedirs(self.gen_dir)

        self.model_save_name = conf.model_name + "_" + conf.active_func + "_" + conf.initial_type
        self.build(conf.model_name, conf.run_type, conf.active_func, conf.initial_type)
        self.train_summary = self.config_summary('train')
        if 'model_GAN' not in self.conf.model_name:
            self.valid_summary = self.config_summary('valid')

    def def_params(self):
        """
            This function make input, output size parameters
        """
        # MNIST data image of shape 28 * 28 = 784
        if 'model_cnn' in self.conf.model_name:
            self.input_shape = [
                None, self.conf.width, self.conf.height, self.conf.channel]
            self.output_shape = [None, self.conf.class_num]
            self.input_x_shape = self.conf.width * self.conf.height * self.conf.channel
        else:
            self.input_shape = [
                None, self.conf.width * self.conf.height * self.conf.channel]
            self.output_shape = [None, self.conf.class_num]
            self.input_x_shape = self.conf.width * self.conf.height * self.conf.channel
        if 'model_GAN' in self.conf.model_name:
            self.output_shape = [None, self.conf.class_num]
            self.input_x_shape = self.conf.width * self.conf.height * self.conf.channel
            self.input_latent_shape = self.conf.num_latent_variable
            self.latent_shape = [None, self.conf.num_latent_variable]

    def build(self, model_type, run_type, active_func, initial_type):
        """
            This function make the placeholder X, Y and Z for GAN.
            It also build dnn, cnn, cnn architecture.
        """
        self.X = tf.placeholder(
            tf.float32, self.input_shape, name='inputs')
        self.Y = tf.placeholder(
            tf.float32, self.output_shape, name='labels')

        if "model_mlp_1" in model_type:
            self.cal_hypothesis(active_func, initial_type)
        elif "model_mlp_3" in model_type:
            self.cal_hypothesis_mlp3(active_func, initial_type)
        elif "model_cnn_1" in model_type:
            self.cal_hypothesis_cnn_1(active_func, initial_type)
        elif "model_cnn_3" in model_type:
            self.cal_hypothesis_cnn_3(active_func, initial_type)
        elif "model_GAN" in model_type:
            self.Z = tf.placeholder(
                tf.float32, self.latent_shape, name='input')
            weight_list = self.cal_hypothesis_D()
            self.G = self.cal_hypothesis_G(active_func, initial_type)

            d_real, d_real_logits = self.build_discriminator(self.X, weight_list)  # D(x)
            d_fake, d_fake_logits = self.build_discriminator(self.G, weight_list)  # D(G(z))

            self.set_gan_cost(d_real_logits, d_fake_logits)
            self.set_gan_optimizer()

        if "model_GAN" not in model_type:
            self.set_cost()
            self.set_optimizer()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if run_type == 2:
            self.writer = None
        else:
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

    def cal_hypothesis_G(self, active_func='sigmoid', initial_type='random'):
        """
         This function build gernerative  model.
         """
        with tf.variable_scope('generator'):
            hidden_layers = self.conf.GAN_hidden.split(',')
            if initial_type == 'random':
                h1 = tf.Variable(tf.random_normal([self.input_latent_shape, int(hidden_layers[0])], stddev=5e-2))
                b1 = tf.Variable(tf.constant(0.1, shape=[int(hidden_layers[0])]))
                h3 = tf.Variable(tf.random_normal([int(hidden_layers[0]), self.input_x_shape], stddev=5e-2))
                b3 = tf.Variable(tf.constant(0.1, shape=[self.input_x_shape]))

            elif initial_type == 'xavier':
                h1 = tf.get_variable("h1", shape=[self.input_latent_shape, int(hidden_layers[0])],
                                     initializer=tf.contrib.layers.xavier_initializer())
                b1 = tf.get_variable("b1", shape=[int(hidden_layers[0])],
                                     initializer=tf.contrib.layers.xavier_initializer())
                h3 = tf.get_variable("h3", shape=[int(hidden_layers[0]), self.input_x_shape],
                                     initializer=tf.contrib.layers.xavier_initializer())
                b3 = tf.get_variable("b3", shape=[self.input_x_shape],
                                     initializer=tf.contrib.layers.xavier_initializer())

            if active_func == 'sigmoid':
                layer_1 = tf.nn.sigmoid(tf.matmul(self.Z, h1) + b1)
            elif active_func == 'relu':
                layer_1 = tf.nn.relu(tf.matmul(self.Z, h1) + b1)

            self.output_layer = tf.matmul(layer_1, h3) + b3
            _generative_hypothesis = tf.nn.sigmoid(self.output_layer)  # generated_mnist_image
        return _generative_hypothesis

    def cal_hypothesis_D(self, initial_type='random'):
        """
        This function build Descriminative model.
        """
        with tf.variable_scope('discriminator'):
            hidden_layers = self.conf.GAN_hidden.split(',')
            weight_list = list()
            if initial_type == 'random':
                dh1 = tf.Variable(tf.random_normal([self.input_x_shape, int(hidden_layers[0])], stddev=5e-2))
                db1 = tf.Variable(tf.constant(0.1, shape=[int(hidden_layers[0])]))

                dh2 = tf.Variable(tf.random_normal([int(hidden_layers[0]), 1], stddev=5e-2))
                db2 = tf.Variable(tf.constant(0.1, shape=[1]))

            elif initial_type == 'xavier':
                dh1 = tf.get_variable("h1", shape=[self.input_x_shape, int(hidden_layers[0])],
                                      initializer=tf.contrib.layers.xavier_initializer())
                db1 = tf.get_variable("b1", shape=[int(hidden_layers[0])],
                                      initializer=tf.contrib.layers.xavier_initializer())
                dh2 = tf.get_variable("h3", shape=[int(hidden_layers[0]), 1],
                                      initializer=tf.contrib.layers.xavier_initializer())
                db2 = tf.get_variable("b3", shape=[1],
                                      initializer=tf.contrib.layers.xavier_initializer())
        weight_list.append(dh1)
        weight_list.append(db1)
        weight_list.append(dh2)
        weight_list.append(db2)
        return weight_list

    def build_discriminator(self, _X, weight_list):
        hidden_layer = tf.nn.relu((tf.matmul(_X, weight_list[0]) + weight_list[1]))
        _logits = tf.matmul(hidden_layer, weight_list[2]) + weight_list[3]  # hy[er]
        predicted_value = tf.nn.sigmoid(_logits)

        return predicted_value, _logits

    def set_gan_cost(self, D_real_logits, D_fake_logits):

        # Discriminator의 손실 함수를 정의합니다.
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(
            D_real_logits)))  # log(D(x))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(
            D_fake_logits)))  # log(1-D(G(z)))
        self.d_loss = d_loss_real + d_loss_fake  # log(D(x)) + log(1-D(G(z)))

        # Generator의 손실 함수를 정의합니다.
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(
            D_fake_logits)))  # log(D(G(z))

    def set_cost(self):
        with tf.variable_scope('loss/loss_op'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.output_layer, labels=self.Y))
        with tf.variable_scope('accuracy'):
            acc = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
            self.accuracy_op = tf.reduce_mean(tf.cast(acc, tf.float32))

    def set_optimizer(self, opti_type=2):

        if opti_type == 1:
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.conf.learning_rate).minimize(
                self.cost, name='train_op')
        else:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.conf.learning_rate).minimize(
                self.cost, name='train_op')

    def set_gan_optimizer(self):
        #
        tvar = tf.trainable_variables()
        dvar = [var for var in tvar if 'discriminator' in var.name]
        gvar = [var for var in tvar if 'generator' in var.name]

        # Discriminator와 Generator의 Optimizer를 정의합니다.
        self.d_train_step = tf.train.AdamOptimizer(self.conf.learning_rate).minimize(self.d_loss, var_list=dvar)
        self.g_train_step = tf.train.AdamOptimizer(self.conf.learning_rate).minimize(self.g_loss, var_list=gvar)

    def cal_hypothesis(self, active_func='sigmoid', initial_type='random'):
        """
        This function build mlp 1 layer network.
        """
        if initial_type == 'random':
            W = tf.Variable(tf.random_normal([self.input_x_shape, self.conf.class_num]))
            b = tf.Variable(tf.random_normal([self.conf.class_num]))
        elif initial_type == 'xavier':
            W = tf.get_variable("W", shape=[self.input_x_shape, self.conf.class_num],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", shape=[self.conf.class_num],
                                initializer=tf.contrib.layers.xavier_initializer())

        if active_func == 'sigmoid':
            self.output_layer = tf.nn.sigmoid(tf.matmul(self.X, W) + b)
        elif active_func == 'relu':
            self.output_layer = tf.nn.relu(tf.matmul(self.X, W) + b)
        self.hypothesis = tf.nn.softmax(self.output_layer)

    def cal_hypothesis_mlp3(self, active_func='sigmoid', initial_type='random'):
        """
        This function build mlp 3 layer network.
        """
        hidden_layers = self.conf.mlp_hidden.split(',')
        if initial_type == 'random':
            h1 = tf.Variable(tf.random_normal([self.input_x_shape, int(hidden_layers[0])]))
            b1 = tf.Variable(tf.random_normal([int(hidden_layers[0])]))

            h2 = tf.Variable(tf.random_normal([int(hidden_layers[0]), int(hidden_layers[1])]))
            b2 = tf.Variable(tf.random_normal([int(hidden_layers[1])]))

            h3 = tf.Variable(tf.random_normal([int(hidden_layers[1]), int(hidden_layers[2])]))
            b3 = tf.Variable(tf.random_normal([int(hidden_layers[2])]))

            W = tf.Variable(tf.random_normal([int(hidden_layers[2]), self.conf.class_num]))
            b = tf.Variable(tf.random_normal([self.conf.class_num]))
        elif initial_type == 'xavier':
            h1 = tf.get_variable("h1", shape=[self.input_x_shape, int(hidden_layers[0])],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable("b1", shape=[int(hidden_layers[0])],
                                 initializer=tf.contrib.layers.xavier_initializer())
            h2 = tf.get_variable("h2", shape=[int(hidden_layers[0]), int(hidden_layers[1])],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable("b2", shape=[int(hidden_layers[1])],
                                 initializer=tf.contrib.layers.xavier_initializer())
            h3 = tf.get_variable("h3", shape=[int(hidden_layers[1]), int(hidden_layers[2])],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable("b3", shape=[int(hidden_layers[2])],
                                 initializer=tf.contrib.layers.xavier_initializer())

            W = tf.get_variable("W", shape=[int(hidden_layers[2]), self.conf.class_num],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", shape=[self.conf.class_num],
                                initializer=tf.contrib.layers.xavier_initializer())

        if active_func == 'sigmoid':
            layer_1 = tf.nn.sigmoid(tf.matmul(self.X, h1) + b1)
            layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, h2) + b2)
            layer_3 = tf.nn.sigmoid(tf.matmul(layer_2, h3) + b3)
        elif active_func == 'relu':
            layer_1 = tf.nn.relu(tf.matmul(self.X, h1) + b1)
            layer_2 = tf.nn.relu(tf.matmul(layer_1, h2) + b2)
            layer_3 = tf.nn.relu(tf.matmul(layer_2, h3) + b3)

        self.output_layer = tf.matmul(layer_3, W) + b
        self.hypothesis = tf.nn.softmax(self.output_layer)

    # def cal_hypothesis_cnn(self):
    #     cnn_filter = self.conf.cnn_filter.split(',')
    #     hidden_layers = self.conf.mlp_hidden.split(',')
    #     # 5x5 conv.
    #     h1 = tf.Variable(tf.truncated_normal([5, 5, 1, int(cnn_filter[0])], stddev=0.1))
    #     b1 = tf.Variable(tf.truncated_normal([int(cnn_filter[0])], stddev=0.1))
    #     # 3x3 conv.
    #     h2 = tf.Variable(tf.truncated_normal([3, 3, int(cnn_filter[0]), int(cnn_filter[1])], stddev=0.1))
    #     b2 = tf.Variable(tf.truncated_normal([int(cnn_filter[1])], stddev=0.1))
    #     # 3x3 conv.
    #     h3 = tf.Variable(tf.truncated_normal([3, 3, int(cnn_filter[1]), int(cnn_filter[2])], stddev=0.1))
    #     b3 = tf.Variable(tf.truncated_normal([int(cnn_filter[2])], stddev=0.1))
    #     # fully connected layer
    #     h4 = tf.Variable(tf.truncated_normal([7 * 7 * int(cnn_filter[2]), int(hidden_layers[-1])], stddev=0.1))
    #     b4 = tf.Variable(tf.truncated_normal([int(hidden_layers[-1])], stddev=0.1))
    #     # output softmax layer (10 digits)
    #     h5 = tf.Variable(tf.truncated_normal([int(hidden_layers[-1]), 10], stddev=0.1))
    #     b5 = tf.Variable(tf.truncated_normal([10], stddev=0.1))
    #
    #     stride = 1  # output is 28x28
    #     layer_1 = tf.nn.relu(tf.nn.conv2d(self.X, h1, strides=[1, stride, stride, 1], padding='SAME') + b1)
    #
    #     k = 2  # max pool filter size and stride, will reduce input by factor of 2
    #     layer_2 = tf.nn.relu(tf.nn.conv2d(layer_1, h2, strides=[1, stride, stride, 1], padding='SAME') + b2)
    #     layer_2 = tf.nn.max_pool(layer_2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    #
    #     layer_3 = tf.nn.relu(tf.nn.conv2d(layer_2, h3, strides=[1, stride, stride, 1], padding='SAME') + b3)
    #     layer_3 = tf.nn.max_pool(layer_3, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    #
    #     # reshape the output from the third convolution for the fully connected layer
    #     flat = tf.reshape(layer_3, shape=[-1, 7 * 7 * int(cnn_filter[2])])
    #     fc1 = tf.nn.relu(tf.matmul(flat, h4) + b4)
    #
    #     self.output_layer = tf.matmul(fc1, h5) + b5
    #     self.hypothesis = tf.nn.softmax(self.output_layer)

    def cal_hypothesis_cnn_3(self, active_func='sigmoid', initial_type='random'):
        """
        This function build cnn 3 network.
        """
        cnn_filter = self.conf.cnn_filter.split(',')
        hidden_layers = self.conf.mlp_hidden.split(',')

        if initial_type == 'random':
            # 5x5 conv. window, 1 input channel (gray images), C1 - outputs
            h1 = tf.Variable(tf.random_normal([5, 5, 1, int(cnn_filter[0])]))
            b1 = tf.Variable(tf.random_normal([int(cnn_filter[0])]))
            # 3x3 conv. window, C1 input channels(output from previous conv. layer ), C2 - outputs
            h2 = tf.Variable(tf.random_normal([3, 3, int(cnn_filter[0]), int(cnn_filter[1])]))
            b2 = tf.Variable(tf.random_normal([int(cnn_filter[1])]))
            # 3x3 conv. window, C2 input channels(output from previous conv. layer ), C3 - outputs
            h3 = tf.Variable(tf.random_normal([3, 3, int(cnn_filter[1]), int(cnn_filter[2])]))
            b3 = tf.Variable(tf.random_normal([int(cnn_filter[2])]))
            # fully connected layer, we have to reshpe previous output to one dim,
            # we have two max pool operation in our network design, so our initial size 28x28 will be reduced 2*2=4
            # each max poll will reduce size by factor of 2
            h4 = tf.Variable(tf.random_normal([7 * 7 * int(cnn_filter[2]), int(hidden_layers[-1])]))
            b4 = tf.Variable(tf.random_normal([int(hidden_layers[-1])]))
            # output softmax layer (10 digits)
            h5 = tf.Variable(tf.random_normal([int(hidden_layers[-1]), 10]))
            b5 = tf.Variable(tf.random_normal([10]))
        elif initial_type == 'xavier':
            # 5x5 conv.
            h1 = tf.get_variable("h1", shape=[5, 5, 1, int(cnn_filter[0])],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable("b1", shape=[int(cnn_filter[0])],
                                 initializer=tf.contrib.layers.xavier_initializer())
            # 3x3 conv.
            h2 = tf.get_variable("h2", shape=[3, 3, int(cnn_filter[0]), int(cnn_filter[1])],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable("b2", shape=[int(cnn_filter[1])],
                                 initializer=tf.contrib.layers.xavier_initializer())
            # 3x3 conv.
            h3 = tf.get_variable("h3", shape=[3, 3, int(cnn_filter[1]), int(cnn_filter[2])],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable("b3", shape=[int(cnn_filter[2])],
                                 initializer=tf.contrib.layers.xavier_initializer())
            # fully connected layer, we have to reshpe previous output to one dim,
            h4 = tf.get_variable("h4", shape=[7 * 7 * int(cnn_filter[2]), int(hidden_layers[-1])],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.get_variable("b4", shape=[int(hidden_layers[-1])],
                                 initializer=tf.contrib.layers.xavier_initializer())
            # output softmax layer (10 digits)
            h5 = tf.get_variable("h5", shape=[int(hidden_layers[-1]), 10],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.get_variable("b5", shape=[10],
                                 initializer=tf.contrib.layers.xavier_initializer())
        if active_func == 'sigmoid':
            stride = 1
            layer_1 = tf.nn.sigmoid(tf.nn.conv2d(self.X, h1, strides=[1, stride, stride, 1], padding='SAME') + b1)
            k = 2  # max pool filter size
            layer_2 = tf.nn.sigmoid(tf.nn.conv2d(layer_1, h2, strides=[1, stride, stride, 1], padding='SAME') + b2)
            layer_2 = tf.nn.max_pool(layer_2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
            layer_3 = tf.nn.sigmoid(tf.nn.conv2d(layer_2, h3, strides=[1, stride, stride, 1], padding='SAME') + b3)
            layer_3 = tf.nn.max_pool(layer_3, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
            # reshape the output from the third convolution for the fully connected layer
            flat = tf.reshape(layer_3, shape=[-1, 7 * 7 * int(cnn_filter[2])])
            fc1 = tf.nn.sigmoid(tf.matmul(flat, h4) + b4)
        elif active_func == 'relu':
            stride = 1
            layer_1 = tf.nn.relu(tf.nn.conv2d(self.X, h1, strides=[1, stride, stride, 1], padding='SAME') + b1)
            k = 2  # max pool filter size
            layer_2 = tf.nn.relu(tf.nn.conv2d(layer_1, h2, strides=[1, stride, stride, 1], padding='SAME') + b2)
            layer_2 = tf.nn.max_pool(layer_2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
            layer_3 = tf.nn.relu(tf.nn.conv2d(layer_2, h3, strides=[1, stride, stride, 1], padding='SAME') + b3)
            layer_3 = tf.nn.max_pool(layer_3, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
            # reshape the output from the third convolution for the fully connected layer
            flat = tf.reshape(layer_3, shape=[-1, 7 * 7 * int(cnn_filter[2])])
            fc1 = tf.matmul(flat, h4) + b4

        self.output_layer = tf.matmul(fc1, h5) + b5
        self.hypothesis = tf.nn.softmax(self.output_layer)

    def cal_hypothesis_cnn_1(self, active_func='sigmoid', initial_type='random'):
        """
        This function  build cnn 1 layer network.
        """
        cnn_filter = self.conf.cnn_filter.split(',')
        hidden_layers = self.conf.mlp_hidden.split(',')

        if initial_type == 'random':
            # 5x5 conv.
            h1 = tf.Variable(tf.random_normal([5, 5, 1, int(cnn_filter[0])]))
            b1 = tf.Variable(tf.random_normal([int(cnn_filter[0])]))
            # fully connected layer, we have to reshpe previous output to one dim,
            h4 = tf.Variable(tf.random_normal([14 * 14 * int(cnn_filter[0]), int(hidden_layers[-1])]))
            b4 = tf.Variable(tf.random_normal([int(hidden_layers[-1])]))
            # output softmax layer (10 digits)
            h5 = tf.Variable(tf.random_normal([int(hidden_layers[-1]), 10]))
            b5 = tf.Variable(tf.random_normal([10]))
        elif initial_type == 'xavier':
            # 5x5 conv.
            h1 = tf.get_variable("h1", shape=[5, 5, 1, int(cnn_filter[0])],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable("b1", shape=[int(cnn_filter[0])],
                                 initializer=tf.contrib.layers.xavier_initializer())
            # fully connected layer, we have to reshpe previous output to one dim,
            h4 = tf.get_variable("h4", shape=[14 * 14 * int(cnn_filter[0]), int(hidden_layers[-1])],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.get_variable("b4", shape=[int(hidden_layers[-1])],
                                 initializer=tf.contrib.layers.xavier_initializer())
            # output softmax layer (10 digits)
            h5 = tf.get_variable("h5", shape=[int(hidden_layers[-1]), 10],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.get_variable("b5", shape=[10],
                                 initializer=tf.contrib.layers.xavier_initializer())
        stride = 1
        k = 2
        if active_func == 'sigmoid':
            layer_1 = tf.nn.sigmoid(tf.nn.conv2d(self.X, h1, strides=[1, stride, stride, 1], padding='SAME') + b1)
            layer_1 = tf.nn.max_pool(layer_1, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
        elif active_func == 'relu':
            layer_1 = tf.nn.relu(tf.nn.conv2d(self.X, h1, strides=[1, stride, stride, 1], padding='SAME') + b1)
            layer_1 = tf.nn.max_pool(layer_1, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

        # reshape the output from the third convolution for the fully connected layer
        flat = tf.reshape(layer_1, shape=[-1, 14 * 14 * int(cnn_filter[0])])
        fc1 = tf.nn.sigmoid(tf.matmul(flat, h4) + b4)

        self.output_layer = tf.matmul(fc1, h5) + b5
        self.hypothesis = tf.nn.softmax(self.output_layer)

    def config_summary(self, name):
        """
        This function set summary for tensorboard.
        """

        if "model_GAN" in self.conf.model_name:
            summarys = list()
            summarys.append(tf.summary.scalar(name + '/gloss', self.g_loss))
            summarys.append(tf.summary.scalar(name + '/dloss', self.d_loss))
            summary = tf.summary.merge(summarys)
        else:
            summarys = list()
            summarys.append(tf.summary.scalar(name + '/loss', self.cost))
            summarys.append(tf.summary.scalar(name + '/accuracy', self.accuracy_op))
            summary = tf.summary.merge(summarys)
        return summary

    def set_dataset(self, _dataset):
        """
        This function set dataset from the main method.
        """
        self.dataset = _dataset

    def save_summary(self, summary, step):
        """
        This function saves summary for tensorboard
        """
        self.writer.add_summary(summary, step)

    def train(self):
        """
        This function train the models for mlp cnn
        """
        self.best_val_acc = 0
        self.best_gloval_step = 0
        val_acc = 0
        for epoch in range(self.conf.training_epochs):
            # for accumulation
            avg_cost: int = 0
            # If sample_size=60000 and batch_size=100, then total batch count is 600
            total_batch = int(self.dataset.train.num_examples / self.conf.batch)
            # Per each batch
            for epoch_num in range(total_batch):
                global_step = (total_batch * (epoch + 1)) + epoch_num
                # save tensorboard summary every test_interval numbers.
                if global_step % self.conf.test_interval == 0 or epoch_num == (len(range(total_batch)) - 1):
                    _, summary = self.sess.run([self.cost, self.valid_summary],
                                               feed_dict={self.X: self.dataset.test.images,
                                                          self.Y: self.dataset.test.labels})
                    self.save_summary(summary, global_step)
                    val_acc = self.accuracy_op.eval(session=self.sess,
                                                    feed_dict={self.X: self.dataset.test.images,
                                                               self.Y: self.dataset.test.labels})
                    print("# Validation [{2}/{1}]   --> Accuracy: {0:0.5f}".format(val_acc, global_step, epoch + 1))
                if global_step % self.conf.summary_interval == 0:
                    batch_xs, batch_ys = self.dataset.train.next_batch(self.conf.batch)
                    # Run training (1 batch)
                    c, _, summary = self.sess.run([self.cost, self.optimizer, self.train_summary], feed_dict={
                        self.X: batch_xs, self.Y: batch_ys})
                    self.save_summary(summary, global_step)
                    train_acc = self.accuracy_op.eval(session=self.sess, feed_dict={self.X: batch_xs,
                                                                                    self.Y: batch_ys})
                    print("# Training   [{4}/{3}]   --> Accuracy: {0:0.5f}    Currenct Batch Loss:"
                          " {1:0.5f} / Avg Loss {2:0.5f}".format(train_acc, c, avg_cost, global_step, epoch + 1))
                else:
                    batch_xs, batch_ys = self.dataset.train.next_batch(self.conf.batch)
                    # Run training (1 batch)
                    c, _ = self.sess.run([self.cost, self.optimizer, ], feed_dict={self.X: batch_xs, self.Y: batch_ys})
                    # avg_cost of this epoch
                    avg_cost += c / total_batch

                if self.best_val_acc < val_acc:
                    print("# Save Model [{2}/{0}]   --> Best Accuracy Accuracy: {1:0.5f}".format(
                        global_step, val_acc, epoch + 1))
                    self.save(global_step, val_acc)
                    self.best_val_acc = val_acc
                    self.best_gloval_step = global_step
        return self.best_gloval_step, self.best_val_acc

    def gan_train(self):
        """
        This function train the models for GAN
        """
        for epoch in range(self.conf.training_epochs):
            total_batch = int(self.dataset.train.num_examples / self.conf.batch)
            for epoch_num in range(total_batch):
                global_step = (total_batch * (epoch + 1)) + epoch_num
                # Save image every 5000
                if global_step % self.conf.test_interval == 0 or epoch_num == (len(range(total_batch)) - 1):
                    batch_noise = np.random.uniform(-1., 1., [self.conf.batch, 100])
                    summary = self.sess.run(self.train_summary, feed_dict={self.X: self.dataset.test.images,
                                                                           self.Z: np.random.uniform(-1., 1.,
                                                                                                     [64, 100])})
                    self.save_summary(summary, global_step)

                if global_step % 500 == 0:
                    samples = self.sess.run(self.G, feed_dict={self.Z: np.random.uniform(-1., 1., [64, 100])})
                    fig = plot(samples)
                    plt.savefig(self.conf.gendir + '/%s.png' % str(global_step).zfill(3), bbox_inches='tight'
                                , cmap=plt.get_cmap('gray'))
                    plt.close(fig)
                batch_xs, batch_ys = self.dataset.train.next_batch(self.conf.batch)
                batch_noise = np.random.uniform(-1., 1., [self.conf.batch, 100])
                # d loss
                _, d_loss_print = self.sess.run([self.d_train_step, self.d_loss],
                                                feed_dict={self.X: batch_xs, self.Z: batch_noise})
                # g loss
                _, g_loss_print = self.sess.run([self.g_train_step, self.g_loss], feed_dict={self.Z: batch_noise})
                print("# Training   [{0}/{1}]       Currenct Batch Generative Loss: {2:0.5f} "
                      " Currenct Batch Descriminitive Loss: {3:0.5f}"
                      .format(epoch + 1, global_step, g_loss_print, d_loss_print))
                if global_step % 5000 == 0:
                    print("# Save Model [{0}/{1}]   --> Best loss: {1:0.5f}".format(
                        epoch + 1, global_step))
                    self.save(global_step, g_loss_print)

    def evaluation(self):
        """
        This function evaluation from a test data.
        """
        if 'model_GAN' in self.conf.model_name:
            print("model Gan evaluation is not implemented.")
        else:
            pred_y = self.sess.run([self.hypothesis, ], feed_dict={
                self.X: self.dataset.test.images})

            f1s = f1_score(np.argmax(pred_y[0], axis=1), np.argmax(self.dataset.test.labels, axis=1), average='weighted')
            self.dataset.test.labels

            val_acc = self.accuracy_op.eval(session=self.sess,
                                            feed_dict={self.X: self.dataset.test.images,
                                                       self.Y: self.dataset.test.labels})
            print("# Validation   --> Accuracy: {0:0.5f} F1 Score : {1}".format(val_acc, f1s))

    def predict(self, predict_value):
        """
        This function predict from a value.
        """
        if 'model_GAN' in self.conf.model_name:
            samples = self.sess.run(self.G, feed_dict={self.Z: predict_value})
            return samples
        else:
            prediction = tf.argmax(self.hypothesis, 1)
            return prediction.eval(feed_dict={self.X: [predict_value]}, session=self.sess)

    def save(self, step, acc):
        """
        This function saves the model.
        """
        checkpoint_path = os.path.join(
            self.model_dir, self.model_save_name + "_" + str(acc))
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step, acc):
        """
        This function loads the model.
        """
        model_name = self.model_save_name + "_" + str(acc)
        checkpoint_path = os.path.join(
            self.model_dir, model_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)


def plot(samples, subplot_n=1):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        if subplot_n == 1:
            ax = plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(sample.reshape(28, 28))
    return fig


def imageprepare(argv, type_cnn):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    new_image = Image.new('L', (28, 28), 255)
    if width > height:
        nheight = int(round((20.0 / width * height), 0))
        if nheight == 0:
            nheight = 1
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))
        new_image.paste(img, (4, wtop))
    else:
        nwidth = int(round((20.0 / height * width), 0))
        if nwidth == 0:
            nwidth = 1
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))
        new_image.paste(img, (wleft, 4))
    tv = list(new_image.getdata())
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    if "model_cnn" in type_cnn:
        tva = np.reshape(tva, (28, 28, 1))
    return tva


def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_integer('training_epochs', 150, '# of step for training')
    flags.DEFINE_integer('test_interval', 1000, '# of interval to test a model')
    flags.DEFINE_integer('summary_interval', 100, '# of step to save summary')
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    flags.DEFINE_float('gpu', 1, 'gpu use :1, cpu :0')
    # data
    flags.DEFINE_integer('batch', 100, 'batch size')
    flags.DEFINE_integer('height', 28, 'height size')
    flags.DEFINE_integer('width', 28, 'width size')
    flags.DEFINE_integer('channel', 1, 'channel size')
    flags.DEFINE_integer('num_latent_variable', 100, 'num_latent_variable')
    # Debug
    flags.DEFINE_string('logdir', './logdir', 'Log dir')
    flags.DEFINE_string('modeldir', './modeldir', 'Model dir')
    flags.DEFINE_string('gendir', './gendir', 'Model dir')

    # network architecture
    flags.DEFINE_integer('class_num', 10, 'output class number')
    flags.DEFINE_string('mlp_hidden', '256,256,100,100', 'hidden_layer_number')
    flags.DEFINE_string('GAN_hidden', '128', 'gan_hidden_layer_number')
    flags.DEFINE_string('cnn_filter', '4,8,16', 'hidden_layer_number')
    flags.DEFINE_string('model_name', 'model_GAN',
                        'Choose one [model_mlp_1, model_mlp_3, model_cnn_1, model_cnn_3, model_GAN]')
    flags.DEFINE_string('active_func', 'relu', 'Choose one [sigmoid, relu]')
    flags.DEFINE_string('initial_type', 'xavier', 'Choose one [random, xavier]')

    # Run_Type
    flags.DEFINE_integer('run_type', 2, '1 : train, 2 : predict')
    # fix bug of flags
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):
    conf = configure()
    tfconfig = tf.ConfigProto()
    if conf.gpu == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.3
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if conf.run_type == 1:
        model = Model(tf
                      .Session(config=tfconfig), conf)
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        model.set_dataset(mnist)
        if 'model_cnn' in conf.model_name:
            mnist_cnn = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False)
            model.set_dataset(mnist_cnn)
        if 'model_GAN' in conf.model_name:
            model.gan_train()
        else:
            best_gloval_step, best_val_acc = model.train()
            tf.reset_default_graph()
            model.reload(best_gloval_step, best_val_acc)
            model.evaluation()

    elif conf.run_type == 2:
        with tf.Session(config=tfconfig) as sess:
            conf.model_name = "model_mlp_1"
            conf.active_func = "sigmoid"
            conf.initial_type = "random"
            conf.learning_rate = "0.001"
            model = Model(sess, conf)
            predict_value = imageprepare("./test_9.png", conf.model_name)
            model.reload(26399, 0.9148)
            print("MLP_1 Label : 9      predict {0}".format(model.predict(predict_value)))
        tf.reset_default_graph()
        with tf.Session(config=tfconfig) as sess:
            conf.model_name = "model_mlp_3"
            conf.active_func = "sigmoid"
            conf.initial_type = "random"
            conf.learning_rate = "0.01"
            model = Model(sess, conf)
            predict_value = imageprepare("./test_9.png", conf.model_name)
            model.reload(20349, 0.9715)
            print("MLP_3 Label : 9      predict {0}".format(model.predict(predict_value)))
        tf.reset_default_graph()
        with tf.Session(config=tfconfig) as sess:
            conf.model_name = "model_cnn_3"
            conf.active_func = "relu"
            conf.initial_type = "xavier"
            conf.learning_rate = "0.001"
            model = Model(sess, conf)
            predict_value = imageprepare("./test_9.png", conf.model_name)
            model.reload(15399, 0.99)
            print("MLP_CNN Label : 9      predict {0}".format(model.predict(predict_value)))
        plt.imshow(np.reshape(predict_value, (28, 28)), cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        plt.show()
        tf.reset_default_graph()
        with tf.Session(config=tfconfig) as sess:
            conf.model_name = "model_GAN"
            conf.active_func = "relu"
            conf.initial_type = "xavier"
            conf.learning_rate = "0.001"
            model = Model(sess, conf)
            #predict_value = imageprepare("./test_9.png", conf.model_name)
            model.reload(80000, 2.3016033)
            predict_value = np.random.uniform(-1., 1., [1, 100])
            samples = model.predict(predict_value)
            fig = plot(samples,0)
            #plt.imshow(np.reshape(predict_value, (28, 28)), cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
            plt.show(fig)

    print("end")


if __name__ == '__main__':
    tf.app.run()
