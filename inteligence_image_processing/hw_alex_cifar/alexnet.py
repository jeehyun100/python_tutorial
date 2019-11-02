import tensorflow as tf
import keras


class AlexNet(object):
    def __init__(self, class_count):

        self.class_count = class_count


    def build_model(self, image_height=112, image_width=112, class_count=1000):

        model = keras.models.Sequential()

        # layer 1 - "filters the 224 x 224 x 3 input image with 96 kernels
        #           of size 11 x 11 x 3 with a stride of 4 pixels"
        model.add(keras.layers.Conv2D(filters=96,
                                      kernel_size=(11, 11),
                                      strides=4,
                                      input_shape=(image_height, image_width, 3),
                                      activation="relu",
                                      padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(3, 3),
                                         strides=(2, 2)))

        # layer 2 - "256 kernels of size 5 x 5 x 48"
        model.add(keras.layers.Conv2D(filters=256,
                                      kernel_size=(5, 5),
                                      activation="relu",
                                      padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(3, 3),
                                         strides=(2, 2)))

        # layer 3 - "384 kernels of size 3 x 3 x 256"
        model.add(keras.layers.Conv2D(filters=384,
                                      kernel_size=(3, 3),
                                      activation="relu",
                                      padding="same"))
        # layer 4 - "384 kernels of size 3 x 3 x 192"
        model.add(keras.layers.Conv2D(filters=384,
                                      kernel_size=(3, 3),
                                      activation="relu",
                                      padding="same"))
        # layer 5 - "256 kernels of size 3 x 3 x 192"
        model.add(keras.layers.Conv2D(filters=256,
                                      kernel_size=(3, 3),
                                      activation="relu",
                                      padding="same"))
        model.add(keras.layers.MaxPool2D(pool_size=(3, 3),
                                         strides=(2, 2)))

        # flatten before feeding into FC layers
        model.add(keras.layers.Flatten())

        # fully connected layers
        # "The fully-connected layers have 4096 neurons each."
        # "We use dropout in the first two fully-connected layers..."
        model.add(keras.layers.Dense(units=4096))  # layer 6
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(units=4096))  # layer 7
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(units=class_count))  # layer 8

        # output layer is softmax
        model.add(keras.layers.Activation('softmax'))
        return model
