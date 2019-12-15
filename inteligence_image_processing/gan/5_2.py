import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

total_epoch = 500
batch_size = 100
learning_rate = 0.0002

n_hidden = 256
n_input = 28 * 28
n_noise = 128
lr = 0.0002
train_epoch = 50

class Model():

    def set_param(self, model_type):
        if model_type == 1 or model_type == 3:
            X = tf.placeholder(tf.float32, [None, n_input])
            Z = tf.placeholder(tf.float32, [None, n_noise])
        elif model_type == 2:
            X = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
            Z = tf.placeholder(tf.float32, shape=(None, 1, 1, n_noise))
        return X,Z

    def build(self, model_type, X, Z):
        """
            Build architeciture
        """

        # MNIST data image of shape 28 * 28 = 784
        if model_type == 1:
            return self.one_layer_Gan(X, Z)

        elif model_type==2:
            return self.dc_Gan(X, Z)
        elif model_type == 3:
            return self.multi_layer_Gan(X, Z)

    def one_layer_Gan(self, input_x, input_z):
        def generator( noise_z):
            hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
            output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)
            return output

        def discriminator( inputs):
            hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
            output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)
            return output

        G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
        G_b1 = tf.Variable(tf.zeros([n_hidden]))
        G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
        G_b2 = tf.Variable(tf.zeros([n_input]))

        D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
        D_b1 = tf.Variable(tf.zeros([n_hidden]))
        D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
        D_b2 = tf.Variable(tf.zeros([1]))

        D_var_list = [D_W1, D_b1, D_W2, D_b2]
        G_var_list = [G_W1, G_b1, G_W2, G_b2]

        G = generator(input_z)
        D_gene = discriminator(G)
        D_real = discriminator(input_x)

        loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
        loss_G = tf.reduce_mean(tf.log(D_gene))

        train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
        train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)
        return G, train_D, train_G, D_var_list, G_var_list, loss_D, loss_G

    def multi_layer_Gan(self, input_x, input_z):
        def generator( noise_z):
            hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
            hidden = tf.nn.relu(tf.matmul(hidden, G_W2) + G_b2)
            output = tf.nn.sigmoid(tf.matmul(hidden, G_W3) + G_b3)
            return output

        def discriminator( inputs):
            hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
            hidden = tf.nn.relu(tf.matmul(hidden, D_W2) + D_b2)
            output = tf.nn.sigmoid(tf.matmul(hidden, D_W3) + D_b3)
            return output

        G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
        G_b1 = tf.Variable(tf.zeros([n_hidden]))
        G_W2 = tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev=0.01))
        G_b2 = tf.Variable(tf.zeros([n_hidden]))
        G_W3 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
        G_b3 = tf.Variable(tf.zeros([n_input]))

        D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
        D_b1 = tf.Variable(tf.zeros([n_hidden]))
        D_W2 = tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev=0.01))
        D_b2 = tf.Variable(tf.zeros([n_hidden]))
        D_W3 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
        D_b3 = tf.Variable(tf.zeros([1]))

        D_var_list = [D_W1, D_b1, D_W2, D_b2, D_W3, D_b3]
        G_var_list = [G_W1, G_b1, D_W2, D_b2, G_W3, G_b3]

        G = generator(input_z)
        D_gene = discriminator(G)
        D_real = discriminator(input_x)

        loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
        loss_G = tf.reduce_mean(tf.log(D_gene))

        train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
        train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)
        return G, train_D, train_G, D_var_list, G_var_list, loss_D, loss_G

    def dc_Gan(self, input_x, input_z):
        def generator(x, isTrain=True, reuse=False):
            with tf.variable_scope('generator', reuse=reuse):
                # 1st hidden layer
                conv1 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding='valid')
                lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

                # 2nd hidden layer
                conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
                lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

                # 3rd hidden layer
                conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
                lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

                # 4th hidden layer
                conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
                lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

                # output layer
                conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
                o = tf.nn.tanh(conv5)

                return o

        def discriminator(x, isTrain=True, reuse=False):
            with tf.variable_scope('discriminator', reuse=reuse):
                # 1st hidden layer
                conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
                lrelu1 = lrelu(conv1, 0.2)

                # 2nd hidden layer
                conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
                lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

                # 3rd hidden layer
                conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
                lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

                # 4th hidden layer
                conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
                lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

                # output layer
                conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding='valid')
                o = tf.nn.sigmoid(conv5)

                return o, conv5

        # networks : generator
        G = generator(input_z)

        # networks : discriminator
        D_real, D_real_logits = discriminator(input_x)
        D_fake, D_fake_logits = discriminator(G, reuse=True)

        # loss for each network
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))

        loss_D = D_loss_real + D_loss_fake
        loss_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))

        # trainable variables for each network
        T_vars = tf.trainable_variables()
        D_var_list = [var for var in T_vars if var.name.startswith('discriminator')]
        G_var_list = [var for var in T_vars if var.name.startswith('generator')]

        # optimizer for each network
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_D = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(loss_D, var_list=D_var_list)
            train_G = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(loss_G, var_list=G_var_list)

        return G, train_D, train_G, D_var_list, G_var_list, loss_D, loss_G

def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

def train(conf):
    tfconfig = tf.ConfigProto()
    if conf.gpu == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.6
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gan_model = Model()
    X, Z = gan_model.set_param(conf.model_type)
    G, train_D, train_G, D_var_list, G_var_list, loss_D, loss_G = Model().build(conf.model_type, X, Z)
    sess = tf.Session(config=tfconfig)
    sess.run(tf.global_variables_initializer())

    #preparing train set
    if conf.model_type == 1 or conf.model_type == 3:
        mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
        train_set = mnist.train.images

    elif conf.model_type == 2:
        mnist = input_data.read_data_sets("./mnist/data/", one_hot=True, reshape=[])
        train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval(session = sess)
        #z_noise = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1

    #total_batch = int(mnist.train.num_examples/batch_size)
    loss_val_D, loss_val_G = 0, 0

    for epoch in range(train_epoch):
        for iter in range(mnist.train.num_examples // batch_size):
            # update discriminator
            batch_xs = train_set[iter * batch_size:(iter + 1) * batch_size]
            if conf.model_type == 1 or conf.model_type == 3 :
                noise = get_noise(batch_size, n_noise)
            elif conf.model_type == 2:
                noise = np.random.normal(0, 1, (batch_size, 1, 1, n_noise))

            _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})

            if conf.model_type == 2:
                noise = np.random.normal(0, 1, (batch_size, 1, 1, n_noise))
            _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise, X: batch_xs})

        print('Epoch:', '%04d' % epoch, 'D loss: {:.4}'.format(loss_val_D), 'G loss: {:.4}'.format(loss_val_G))

        if (epoch == 0 or (epoch + 1) % 10 == 0 ) or (conf.model_type == 2):
            sample_size = 10
            if conf.model_type == 1 or conf.model_type == 3:
                noise = get_noise(sample_size, n_noise)
            elif conf.model_type == 2:
                noise = np.random.normal(0, 1, (sample_size, 1, 1, n_noise))

            samples = sess.run(G, feed_dict={Z: noise})

            fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

            for i in range(sample_size):
                ax[i].set_axis_off()
                if conf.model_type == 1 or conf.model_type == 3:
                    ax[i].imshow(np.reshape(samples[i], (28, 28)))
                elif conf.model_type == 2:
                    ax[i].imshow(np.reshape(samples[i], (64, 64)))

            if not os.path.isdir('./samples/'+str(conf.model_type)):
                os.mkdir('./samples/'+str(conf.model_type))

            plt.savefig('./samples/'+str(conf.model_type)+'/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)


def configure():
    # Configure for training
    flags = tf.app.flags
    flags.DEFINE_integer('training_epochs', 150, '# of step for training')
    flags.DEFINE_integer('batch_size', 1000, '# of step for training')
    flags.DEFINE_float('gpu', 1, 'gpu use :1, cpu :0')
    flags.DEFINE_integer('model_type', 2, '1_layer_gan :1,, dc_Gan : 2,  multi_layer_gan : 3')
    #flags.DEFINE_integer('run_type', 1, 'predict :0, transfer_predict : 1, train : 2, transfer_learning : 3,  show_cam : 4, show_filter :5')
    flags.DEFINE_string('additional_img_path', './preprocess/augmentation/', 'addtional images')
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


if __name__ == '__main__':
    conf = configure()
    train(conf)
print('최적화 완료!')


