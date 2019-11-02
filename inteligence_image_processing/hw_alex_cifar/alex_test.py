from __future__ import print_function
import keras
import cv2
import numpy as np
import sys
import os
import argparse
from alexnet import AlexNet
from keras.models import Sequential
from keras import backend as K
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model



def preprocess_image(image, image_height=112, image_width=112):
    """resize images to the appropriate dimensions
    :param image_width:
    :param image_height:
    :param image: image
    :return: image
    """
    return cv2.resize(image, (image_height, image_width))


def load_dataset():
    """loads training and testing resources
    :return: x_train, y_train, x_test, y_test
    """
    return keras.datasets.cifar10.load_data()


def generator(batch_size, class_count, image_height, image_width, x_data, y_data):
    """generates batch training (and evaluating) data and labels
    """

    while True:
        X = []  # batch training set
        Y = []  # batch labels
        for index in range(0, len(x_data)):
            X.append(preprocess_image(x_data[index], image_height, image_width))
            Y.append(y_data[index])
            if (index + 1) % batch_size == 0:
                yield np.array(X), keras.utils.to_categorical(np.array(Y), class_count)
                X = []
                Y = []


def generator_ramdom_one(class_count, image_height, image_width, x_data, y_data):
    """generates batch training (and evaluating) data and labels
    """
    while True:
        X = []  # batch training set
        Y = []  # batch labels

        x = random.randint(0, 10000)
        X.append(preprocess_image(x_data[x], image_height, image_width))
        Y.append(y_data[x])
        yield np.array(X), keras.utils.to_categorical(np.array(Y), class_count)


def train_model(model, image_height=112, image_width=112, class_count=1000, epochs=90):
    """train the SuperVision/alexnet NN model
    :param epochs:
    :param image_height:
    :param class_count:
    :param image_width:
    :param model: NN model (uncompiled, without weights)
    :return: compiled NN model with weights
    """
    # compile with SGD optimizer and categorical_crossentropy as the loss function
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=0.02, momentum=0.9, decay=0.0005),
                  metrics=['accuracy'])

    # training parameters
    (x_train, y_train), (x_test, y_test) = load_dataset()
    batch_size = 128
    steps = len(x_train) / batch_size

    # train the model using a batch generator
    batch_generator = generator(batch_size, class_count, image_height, image_width, x_train, y_train)
    model.fit_generator(generator=batch_generator,
                        steps_per_epoch=steps,
                        epochs=epochs,
                        verbose=1)

    # train the model on the dataset
    # count=10000
    # x_train = np.array([preprocess_image(image) for image in x_train[:count]])
    # y_train = keras.utils.to_categorical(y_train[:count], class_count)
    # model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)


def evaluate(model, class_count=1000, image_height=112, image_width=112):
    """evaluate the performance of the trained model using the prepared testing set
    :param image_width:
    :param class_count:
    :param image_height:
    :param model: compiled NN model with trained weights
    """

    # training parameters
    (x_train, y_train), (x_test, y_test) = load_dataset()
    batch_size = 128
    steps = len(x_test) / batch_size

    # train the model using a batch generator
    batch_generator = generator(batch_size, class_count, image_height, image_width, x_test, y_test)
    scores = model.evaluate_generator(generator=batch_generator,
                                      #verbose=1,
                                      steps=steps)
    print("Test Loss:\t", scores[0])
    print("Test Accuracy:\t", scores[1])


def show_filter(model, img_data, layer_name):

    activation_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    #input_tensor = np.expand_dims(img_tensor, 0)
    layer_activation = activation_model.predict(img_data)

    images_per_row = 16

    n_features = layer_activation.shape[-1]

    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row  # 열의 갯수를 16개의 단위로 보여주고 싶어서입니다.

    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            # 모든 채널을 사용하겠다.
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]
            # 픽셀로 표현하기 위해.
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            # np.clip은 relu같은 느낌입니다. 0밑은 전부 0, 255 위는 전부 255로.
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,
            row * size: (row + 1) * size] = channel_image
    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
    print("done")


def parse_arguments():
    """parse command line input
    :return: dictionary of arguments keywords and values
    """

    default_model_name = 'alexnet_v2.h5'
    default_model_dir = 'models'
    parser = argparse.ArgumentParser(description="Construct and train an alexnet model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n',
                        default=default_model_name,
                        metavar='<model_name>',
                        help='The name to be given to the output model.')
    parser.add_argument('-d',
                        default=default_model_dir,
                        metavar='<output_directory>',
                        help='The directory in which the models should be saved.')
    parser.add_argument('-e',
                        default=25,
                        metavar='<number_of_epochs>',
                        help='The number of epochs used to train the model. The original alexnet used 90 epochs.')
    parser.add_argument('-p',
                        default=False,
                        metavar='<predict flag>',
                        help='Predict flag')
    parser.add_argument('-f',
                        default=False,
                        metavar='<show filter if predict flag True>',
                        help='Predict flag')
    return vars(parser.parse_args())


def main():
    """build, train, and test an implementation of the alexnet CNN model in keras.
    This model is trained and tested on the CIFAR-100 dataset
    """
    # parse arguments
    args = parse_arguments()
    save_dir = os.path.join(os.getcwd(), args['d'])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, args['n'])
    epochs = int(args['e'])
    predict_flag = args['p']
    show_flag = args['f']
    cifar10_label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # build and train the model
    if predict_flag == True:
        if show_flag == True:
            print(filter)
            alexnet = AlexNet(10)
            model = alexnet.build_model(class_count=10)
            model.load_weights(model_path)
            print("load complete")

            (x_train, y_train), (x_test, y_test) = load_dataset()
            batch_generator = generator_ramdom_one(10, 112, 112, x_test, y_test)
            test_x, test_y = next(batch_generator)

            layer = "conv2d_2"
            show_filter(model, test_x, layer)


        else :
            alexnet = AlexNet(10)
            model = alexnet.build_model(class_count=10)
            model.load_weights(model_path)
            print("load complete")

            (x_train, y_train), (x_test, y_test) = load_dataset()
            batch_generator = generator_ramdom_one(10, 112, 112, x_test, y_test)

            test_x, test_y = next(batch_generator)
            print(cifar10_label[np.argmax(test_y)])
            result = model.predict(test_x)
            print(cifar10_label[np.argmax(result)])
            test_img_x = test_x[0, ::-1].copy()
            plt.imshow(test_img_x/255, interpolation='nearest')
            plt.title(cifar10_label[np.argmax(test_y)] +"/" + cifar10_label[np.argmax(result)])
            plt.show()

    else:
        alexnet = AlexNet(10)
        model = alexnet.build_model(class_count=10)
        print(model.summary())
        train_model(model, class_count=10, epochs=epochs)

        # test the model
        evaluate(model, class_count=10)

        # save the trained model
        model.save(model_path)
        print("Alexnet model saved to: %s" % model_path)


if __name__ == "__main__":
    # execute only if run as a script
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)

