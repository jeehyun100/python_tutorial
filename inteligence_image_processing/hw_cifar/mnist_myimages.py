# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from random import shuffle
import matplotlib.image as mpimg
tf.set_random_seed(777)  # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
myimages = np.array([], dtype='f')
mylabels = np.array([], dtype='i')
def rgb_to_gray(img):
    grayImage = np.zeros(img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    R = (R *.299)
    G = (G *.587)
    B = (B *.114)

    Avg = 1-(R+G+B)
    # Avg = (R+G+B)

    Avg = Avg/(np.max(Avg))

    grayImage = Avg

    return grayImage

FileList = []
path = './imgfiles'
for filename in os.listdir(path):
    if filename.endswith(".png"):
        FileList.append(os.path.join(path, filename))
shuffle(FileList)
for filename in FileList:
    image = mpimg.imread(filename)
    label_tmp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='i')
    label = int(filename.split('\\')[1][0])
    label_tmp[label] = 1
    label_tmp = label_tmp.reshape(1,10)
    grayImage = rgb_to_gray(image)
    grayImage = grayImage.reshape(1,784)
    if myimages.shape[0] == 0:
        myimages = np.append(myimages, grayImage)
        myimages = myimages.reshape(1,784)
        mylabels = np.append(mylabels, label_tmp)
        mylabels = mylabels.reshape(1,10)
    else:
        myimages = np.append(myimages, grayImage, axis=0)
        mylabels = np.append(mylabels, label_tmp, axis=0)

nb_hidden1 = 150
# nb_hidden2 = 30
nb_classes = 10
rate_learn = 0.2
w_stdev = 0.1
# w_stdev = 0.036 # 0.036 = 1 / (784)^0.5

# MNIST data image of shape 28 * 28 = 784
X  = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y  = tf.placeholder(tf.float32, [None, nb_classes])

# W1 = tf.get_variable("W1", shape=[784, nb_hidden1], initializer=tf.contrib.layers.xavier_initializer())
W1 = tf.Variable(tf.random_normal([784, nb_hidden1], stddev=w_stdev))
b1 = tf.Variable(tf.zeros([nb_hidden1]))
# W2 = tf.get_variable("W2", shape=[nb_hidden1, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.Variable(tf.random_normal([nb_hidden1, nb_classes], stddev=w_stdev))
b2 = tf.Variable(tf.zeros([nb_classes]))

# Hypothesis (using softmax)
H1         = tf.nn.relu(tf.matmul(X,     W1) + b1)
hypothesis = tf.nn.softmax(tf.matmul(H1, W2) + b2)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate_learn).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate=rate_learn).minimize(cost)

# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
training_epochs = 15
# training_epochs = 15
batch_size = 100

for k in range(1,2):
    start_time = time.time()
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)

            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                c, _ = sess.run([cost, optimizer], feed_dict={
                                X: batch_xs, Y: batch_ys})
                avg_cost += c / total_batch

            print('Epoch:', '%04d' % (epoch + 1),
                  'cost =', '{:.9f}'.format(avg_cost))

        print("Learning finished")

        # Test the model using test sets
        print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
              X: myimages, Y: mylabels}))
              # X: mnist.test.images, Y: mnist.test.labels}))


        # Get one and predict
        r = random.randint(0, mylabels.shape[0] - 1)
        print("Label: ", sess.run(tf.argmax(mylabels[r:r + 1], 1)))
        # r = random.randint(0, mnist.test.num_examples - 1)
        print("Prediction: ", sess.run(
            tf.argmax(hypothesis, 1), feed_dict={X: myimages[r:r + 1]}))
            # tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))



    plt.imshow(
        myimages[r:r + 1].reshape(28, 28),
        # mnist.test.images[r:r + 1].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest')
    plt.show()
    
    # print("start_time", start_time)
    print("--- %s seconds ---" %(time.time() - start_time))