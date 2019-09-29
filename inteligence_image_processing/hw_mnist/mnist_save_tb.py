import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
from PIL import Image, ImageFilter
import os
import numpy as np
import matplotlib.pyplot as plt  # for plotting


class Model(object):

    def __init__(self, sess, conf):
        self.output_layer = None
        self.hypothesis = None
        self.dataset = None
        self.accuracy_op = None
        self.X = None
        self.Y = None
        self.saver = None
        self.writer = None
        self.cost = None
        self.optimizer = None
        self.output_shape = None
        self.input_shape = None
        self.input_x_shape = None

        self.sess = sess
        self.conf = conf
        self.def_params()
        self.model_dir = conf.modeldir + "/" + conf.model_name
        self.log_dir = conf.logdir + "/" + conf.model_name
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.build(conf.model_name, conf.run_type)
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')

    def def_params(self):

        # MNIST data image of shape 28 * 28 = 784
        if self.conf.model_name != 'model_cnn':
            self.input_shape = [
                None, self.conf.width * self.conf.height * self.conf.channel]
            self.output_shape = [None, self.conf.class_num]
            self.input_x_shape = self.conf.width * self.conf.height * self.conf.channel
        else:
            self.input_shape = [
                None, self.conf.width, self.conf.height, self.conf.channel]
            self.output_shape = [None, self.conf.class_num]
            self.input_x_shape = self.conf.width * self.conf.height * self.conf.channel

    def build(self, model_type, run_type):
        self.X = tf.placeholder(
            tf.float32, self.input_shape, name='inputs')
        self.Y = tf.placeholder(
            tf.float32, self.output_shape, name='labels')
        if model_type == "model_mlp_1":
            self.cal_hypothesis()
        elif model_type == "model_mlp_2":
            self.cal_hypothesis_mlp2()
        elif model_type == "model_cnn":
            self.cal_hypothesis_cnn()
        self.set_cost()
        self.set_optimizer()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if run_type == 2:
            self.writer = None
        else:
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

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

    def cal_hypothesis(self):
        W = tf.Variable(tf.random_normal([self.input_x_shape, self.conf.class_num]))
        b = tf.Variable(tf.random_normal([self.conf.class_num]))
        self.output_layer = tf.matmul(self.X, W) + b
        self.hypothesis = tf.nn.softmax(self.output_layer)

    def cal_hypothesis_mlp2(self):
        hidden_layers = self.conf.mlp_hidden.split(',')

        h1 = tf.Variable(tf.random_normal([self.input_x_shape, int(hidden_layers[0])]))
        b1 = tf.Variable(tf.random_normal([int(hidden_layers[0])]))

        h2 = tf.Variable(tf.random_normal([int(hidden_layers[0]), int(hidden_layers[1])]))
        b2 = tf.Variable(tf.random_normal([int(hidden_layers[1])]))

        W = tf.Variable(tf.random_normal([int(hidden_layers[1]), self.conf.class_num]))
        b = tf.Variable(tf.random_normal([self.conf.class_num]))

        layer_1 = tf.matmul(self.X, h1) + b1
        layer_2 = tf.matmul(layer_1, h2) + b2

        self.output_layer = tf.matmul(layer_2, W) + b
        self.hypothesis = tf.nn.softmax(self.output_layer)

    def cal_hypothesis_cnn(self):
        cnn_filter = self.conf.cnn_filter.split(',')
        hidden_layers = self.conf.mlp_hidden.split(',')

        # 5x5 conv. window, 1 input channel (gray images), C1 - outputs
        h1 = tf.Variable(tf.truncated_normal([5, 5, 1, int(cnn_filter[0])], stddev=0.1))
        b1 = tf.Variable(tf.truncated_normal([int(cnn_filter[0])], stddev=0.1))
        # 3x3 conv. window, C1 input channels(output from previous conv. layer ), C2 - outputs
        h2 = tf.Variable(tf.truncated_normal([3, 3, int(cnn_filter[0]), int(cnn_filter[1])], stddev=0.1))
        b2 = tf.Variable(tf.truncated_normal([int(cnn_filter[1])], stddev=0.1))
        # 3x3 conv. window, C2 input channels(output from previous conv. layer ), C3 - outputs
        h3 = tf.Variable(tf.truncated_normal([3, 3, int(cnn_filter[1]), int(cnn_filter[2])], stddev=0.1))
        b3 = tf.Variable(tf.truncated_normal([int(cnn_filter[2])], stddev=0.1))
        # fully connected layer, we have to reshpe previous output to one dim,
        # we have two max pool operation in our network design, so our initial size 28x28 will be reduced 2*2=4
        # each max poll will reduce size by factor of 2
        h4 = tf.Variable(tf.truncated_normal([7 * 7 * int(cnn_filter[2]), int(hidden_layers[-1])], stddev=0.1))
        b4 = tf.Variable(tf.truncated_normal([int(hidden_layers[-1])], stddev=0.1))

        # output softmax layer (10 digits)
        h5 = tf.Variable(tf.truncated_normal([int(hidden_layers[-1]), 10], stddev=0.1))
        b5 = tf.Variable(tf.truncated_normal([10], stddev=0.1))

        stride = 1  # output is 28x28
        layer_1 = tf.nn.relu(tf.nn.conv2d(self.X, h1, strides=[1, stride, stride, 1], padding='SAME') + b1)

        k = 2  # max pool filter size and stride, will reduce input by factor of 2
        layer_2 = tf.nn.relu(tf.nn.conv2d(layer_1, h2, strides=[1, stride, stride, 1], padding='SAME') + b2)
        layer_2 = tf.nn.max_pool(layer_2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

        layer_3 = tf.nn.relu(tf.nn.conv2d(layer_2, h3, strides=[1, stride, stride, 1], padding='SAME') + b3)
        layer_3 = tf.nn.max_pool(layer_3, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

        # reshape the output from the third convolution for the fully connected layer
        flat = tf.reshape(layer_3, shape=[-1, 7 * 7 * int(cnn_filter[2])])
        fc1 = tf.nn.relu(tf.matmul(flat, h4) + b4)

        self.output_layer = tf.matmul(fc1, h5) + b5
        self.hypothesis = tf.nn.softmax(self.output_layer)

    def config_summary(self, name):
        summarys = list()
        summarys.append(tf.summary.scalar(name+'/loss', self.cost))
        summarys.append(tf.summary.scalar(name+'/accuracy', self.accuracy_op))
        summary = tf.summary.merge(summarys)
        return summary

    def set_dataset(self, dataset):
        """
        This function set dataset from the main method.
        """
        self.dataset = dataset

    def save_summary(self, summary, step):
        """
        This function saves summary for tensorboard
        """
        self.writer.add_summary(summary, step)

    def train(self):
        """
        This function train the models
            every test_interval validate from test data.
            every summary_interval save tensorboard info

        """
        best_val_acc = 0
        val_acc = 0
        #global_step = 0
        for epoch in range(self.conf.training_epochs):
            # for accumulation
            avg_cost: int = 0
            # If sample_size=60000 and batch_size=100, then total batch count is 600
            total_batch = int(self.dataset.train.num_examples / self.conf.batch)
            # Per each batch
            for epoch_num in range(total_batch):
                global_step = (total_batch * (epoch + 1)) + epoch_num
                # save tensorboard summary every test_interval numbers.
                if global_step % self.conf.test_interval == 0:
                    _, summary = self.sess.run([self.cost, self.valid_summary],
                                               feed_dict={self.X: self.dataset.test.images,
                                                          self.Y: self.dataset.test.labels})
                    self.save_summary(summary, global_step)
                    val_acc = self.accuracy_op.eval(session=self.sess,
                                                    feed_dict={self.X: self.dataset.test.images,
                                                               self.Y: self.dataset.test.labels})
                    print("# Validation [{2}/{1}]   --> Accuracy: {0:0.5f}".format(val_acc, epoch_num, epoch+1))
                if global_step % self.conf.summary_interval == 0:
                    batch_xs, batch_ys = self.dataset.train.next_batch(self.conf.batch)
                    # Run training (1 batch)
                    c, _, summary = self.sess.run([self.cost, self.optimizer, self.train_summary], feed_dict={
                        self.X: batch_xs, self.Y: batch_ys})
                    self.save_summary(summary, global_step)
                    train_acc = self.accuracy_op.eval(session=self.sess, feed_dict={self.X: batch_xs,
                                                      self.Y: batch_ys})

                    print("# Training   [{4}/{3}]   --> Accuracy: {0:0.5f}    Currenct Batch Loss:"
                          " {1:0.5f} / Avg Loss {2:0.5f}".format(train_acc, c, avg_cost, epoch_num, epoch+1))
                else:
                    batch_xs, batch_ys = self.dataset.train.next_batch(self.conf.batch)
                    # Run training (1 batch)
                    c, _ = self.sess.run([self.cost, self.optimizer, ], feed_dict={self.X: batch_xs, self.Y: batch_ys})
                    # avg_cost of this epoch
                    avg_cost += c / total_batch

                if best_val_acc < val_acc:
                    print("# Save Model [{2}/{0}]   --> Best Accuracy Accuracy: {1:0.5f}".format(
                        epoch_num, val_acc, epoch+1))
                    self.save(global_step, val_acc)
                    best_val_acc = val_acc

    def predict(self, predict_value):
        """
        This function predict from a value.
        """

        prediction = tf.argmax(self.hypothesis, 1)
        return prediction.eval(feed_dict={self.X: [predict_value]}, session=self.sess)

    def save(self, step, acc):
        """
        This function saves the model.
        """
        checkpoint_path = os.path.join(
            self.model_dir, self.conf.model_name+"_"+str(acc))
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step, acc):
        """
        This function loads the model.
        """
        model_name = self.conf.model_name + "_"+str(acc)
        checkpoint_path = os.path.join(
            self.model_dir, model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)


def imageprepare(argv, type_cnn):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), 255)  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if nheight == 0:  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if nwidth == 0:  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    if type_cnn == "model_cnn":
        tva = np.reshape(tva, (28, 28, 1))
    return tva


def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_integer('training_epochs', 25, '# of step for training')
    flags.DEFINE_integer('test_interval', 1000, '# of interval to test a model')
    flags.DEFINE_integer('summary_interval', 100, '# of step to save summary')
    flags.DEFINE_float('learning_rate', 0.003, 'learning rate')
    # data
    flags.DEFINE_integer('batch', 100, 'batch size')
    flags.DEFINE_integer('height', 28, 'height size')
    flags.DEFINE_integer('width', 28, 'width size')
    flags.DEFINE_integer('channel', 1, 'channel size')
    # Debug
    flags.DEFINE_string('logdir', './logdir', 'Log dir')
    flags.DEFINE_string('modeldir', './modeldir', 'Model dir')
    # network architecture
    flags.DEFINE_integer('class_num', 10, 'output class number')
    flags.DEFINE_string('mlp_hidden', '256,256', 'hidden_layer_number')
    flags.DEFINE_string('cnn_filter', '4,8,16', 'hidden_layer_number')
    flags.DEFINE_string('model_name', 'model_mlp_1', 'Choose one [model_mlp_1, model_mlp_2, model_cnn]')
    # Run_Type
    flags.DEFINE_integer('run_type', 2, '1 : train, 2 : predict')
    # fix bug of flags
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):

    # args = parser.parse_args()
    conf = configure()

    if conf.run_type == 1:
        model = Model(tf.Session(), conf)
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        if conf.model_name == 'model_cnn':
            mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False)
        model.set_dataset(mnist)
        model.train()
    elif conf.run_type == 2:
        with tf.Session() as sess:
            conf.model_name = "model_mlp_1"
            model = Model(sess, conf)
            predict_value = imageprepare("./test_9.png", conf.model_name)
            model.reload(13000, 0.9183)
            print("MLP_1 Label : 9      predict {0}".format(model.predict(predict_value)))

        tf.reset_default_graph()
        with tf.Session() as sess:
            conf.model_name = "model_mlp_2"
            model = Model(sess, conf)
            predict_value = imageprepare("./test_9.png", conf.model_name)
            model.reload(5000, 0.8915)
            print("MLP_2 Label : 9      predict {0}".format(model.predict(predict_value)))
        tf.reset_default_graph()
        with tf.Session() as sess:
            conf.model_name = "model_cnn"
            model = Model(sess, conf)
            predict_value = imageprepare("./test_9.png", conf.model_name)
            model.reload(6000, 0.9906)
            print("MLP_CNN Label : 9      predict {0}".format(model.predict(predict_value)))
        plt.imshow(np.reshape(predict_value,(28,28)), cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        plt.show()

    print("end")


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    tf.app.run()
