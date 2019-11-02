from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import cv2
from PIL import Image as pil_image
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
import os
import glob
import random


class cifar10vgg:
    def __init__(self, train=0):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.33
        set_session(tf.Session(config=config))

        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32, 32, 3]

        self.model = self.build_model()
        self.transfer_model_name = 'cifar10vgg_transfer_only_cnn2_fc.h5'
        if train == 1:
            self.model = self.train(self.model)
        elif train == 2:
            self.model.load_weights('cifar10vgg.h5')
            self.model = self.transfer_running(self.model, self.transfer_model_name)
        elif train == 0:
            self.model = self.transfer_predict(self.model)
            self.model.load_weights(self.transfer_model_name)
            # print("1")
        elif train == 3:
            self.model.load_weights('cifar10vgg.h5')

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model

    def normalize(self, X_train, X_test):
        # this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test_img set
        # Output: normalized training set and test_img set according to the trianing set statistics.
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
        return X_train, X_test

    def normalize_production(self, x):
        # this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        # these values produced during first training and are general for the standard cifar10 training set normalization
        # for transfer
        mean = 120.06
        std = 63.41
        return (x - mean) / (std + 1e-7)

    def predict(self, x, normalize=True, batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x, batch_size)

    def train(self, model):

        # training parameters
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20
        # The data, shuffled and split between train and test_img sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        # data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # training process in a for loop with learning rate drop every 25 epoches.

        historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                                       batch_size=batch_size),
                                          steps_per_epoch=x_train.shape[0] // batch_size,
                                          epochs=maxepoches,
                                          validation_data=(x_test, y_test), callbacks=[reduce_lr], verbose=2)
        model.save_weights('cifar10vgg.h5')
        return model

    def load_addtional_data(self, image_path):
        batch = []
        for file in glob.glob(image_path + "train/" + '*.png'):
            try:
                _img = cv2.imread(file)
                img = _img[..., ::-1].copy()
            except FileExistsError as e:
                None
            batch.append(img)
        all_images = np.array(batch)
        all_labels = np.empty((all_images.shape[0], 1))
        all_labels.fill(10)
        # all_labels = np.expand_dims(all_labels, 1)
        # total_row_number = all_images.shape[0]
        # test_sample_number = int(total_row_number*0.1)
        np.random.shuffle(all_images)

        batch_test = []
        for file in glob.glob(image_path + "test/" + '*.png'):
            try:
                _img = cv2.imread(file)
                img = _img[..., ::-1].copy()
            except FileExistsError as e:
                None
            batch_test.append(img)
        test_images = np.array(batch_test)
        test_labels = np.empty((test_images.shape[0], 1))
        test_labels.fill(10)

        return (all_images, all_labels), (test_images, test_labels)

    def transfer_predict(self, model):

        layer_name = 'dropout_9'
        weight_decay = self.weight_decay

        drop_fn_model = Model(inputs=self.model.input, outputs=model.model.get_layer(layer_name).output)

        additional_class = 11
        # Adding custom Layers
        x = drop_fn_model.output

        # x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        # x = Activation('relu')(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.4)(x)
        #
        #
        # x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        # x = Activation('relu')(x)
        # x = BatchNormalization()(x)
        #
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(512)(x)
        x = (Activation('relu'))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(additional_class)(x)
        prediction = Activation('softmax')(x)
        #
        new_fn_model = Model(inputs=drop_fn_model.input, outputs=prediction)
        return new_fn_model

    def transfer_running(self, model, model_name):
        # ff = tf.app.flags.FLAGS
        # additional_img_path = tf.app.flags.FLAGS.additional_img_path
        # additional_images = self.load_addtional_data(additional_img_path)
        additional_img_path = tf.app.flags.FLAGS.additional_img_path

        layer_name = 'dropout_9'
        # training parameters
        batch_size = 128
        maxepoches = tf.app.flags.FLAGS.training_epochs
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20
        weight_decay = self.weight_decay

        for layer in model.layers[:44]:
            layer.trainable = False
        drop_fn_model = Model(inputs=self.model.input, outputs=model.model.get_layer(layer_name).output)
        #
        # model.add(Flatten())
        # model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization())
        #
        # model.add(Dropout(0.5))
        # model.add(Dense(self.num_classes))
        # model.add(Activation('softmax'))
        additional_class = 11
        # Adding custom Layers
        x = drop_fn_model.output

        # x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        # x = Activation('relu')(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.4)(x)
        #
        #
        # x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        # x = Activation('relu')(x)
        # x = BatchNormalization()(x)
        #
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = (Activation('relu'))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(additional_class)(x)
        prediction = Activation('softmax')(x)
        #
        new_fn_model = Model(inputs=drop_fn_model.input, outputs=prediction)
        # The data, shuffled and split between train and test_img sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        (x_train_add, y_train_add), (x_test_add, y_test_add) = self.load_addtional_data(additional_img_path)

        x_train_add_raw_num = int(x_train_add.shape[0])
        # x_test_add_raw_num = int(x_test.shape[0])
        # x_train = x_train[:x_train_add_raw_num]
        # y_train = y_train[:x_train_add_raw_num]

        # x_train = x_train.astype('float32')
        # x_test = x_test.astype('float32')
        x_train_hstack = np.vstack((x_train_add, x_train))
        y_train_hstack = np.vstack((y_train_add, y_train))

        x_test_hstack = np.vstack((x_test_add, x_test))
        y_test_hstack = np.vstack((y_test_add, y_test))

        x_train_hstack = x_train_hstack.astype('float32')
        x_test_hstack = x_test_hstack.astype('float32')

        x_train_hstack, x_test_hstack = self.normalize(x_train_hstack, x_test_hstack)
        # fig, axes = plt.subplots(1, 2, figsize=(8, 6))
        # arr_img1 = np.asarray(x_train_add)[0,:, :, :3] / 255.
        # # axes[0, 1].imshow(x_train[0])
        # # axes[0, 2].imshow(x_train_add[0])
        # plt.imshow(arr_img1)
        # #p#lt.imshow(display_grid, aspect='auto', cmap='viridis')
        # #
        # plt.show()
        # additional_class = 11
        additional_class = 11
        y_train_hstack = keras.utils.to_categorical(y_train_hstack, additional_class)
        y_test_hstack = keras.utils.to_categorical(y_test_hstack, additional_class)

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        # data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train_hstack)

        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        new_fn_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # training process in a for loop with learning rate drop every 25 epoches.

        historytemp = new_fn_model.fit_generator(datagen.flow(x_train_hstack, y_train_hstack,
                                                              batch_size=batch_size),
                                                 steps_per_epoch=x_train_hstack.shape[0] // batch_size,
                                                 epochs=maxepoches,
                                                 validation_data=(x_test_hstack, y_test_hstack), callbacks=[reduce_lr],
                                                 verbose=2)
        new_fn_model.save_weights(model_name)
        return new_fn_model


def preprocess_input(img_path):
    img = pil_image.open(img_path).resize((224, 224))
    img_arr = np.asarray(img)[:, :, :3] / 255.
    img_tensor = np.expand_dims(img_arr, 0)

    return img_arr, img_tensor


def generate_grad_cam_test(img_tensor, model, class_index, activation_layer):
    """
    params:
    -------
    img_tensor: resnet50 모델의 이미지 전처리를 통한 image tensor
    model: pretrained resnet50 모델 (include_top=True)
    class_index: 이미지넷 정답 레이블
    activation_layer: 시각화하려는 레이어 이름

    return:
    grad_cam: grad_cam 히트맵
    """

    img_arr = np.asarray(img_tensor)[0, :, :, :3] / 255.
    # img_arr = np.asarray(img_tensor)[0, :, :, :3]
    # img_tensor = img_arr#np.expand_dims(img_arr, 0)

    inp = model.input
    y_c = model.output.op.inputs[0][0, class_index]
    A_k = model.get_layer(activation_layer).output

    ## 이미지 텐서를 입력해서
    ## 해당 액티베이션 레이어의 아웃풋(a_k)과
    ## 소프트맥스 함수 인풋의 a_k에 대한 gradient를 구한다.
    get_output = K.function([inp], [A_k, K.gradients(y_c, A_k)[0], model.output])
    [conv_output, grad_val, model_output] = get_output([img_tensor])

    ## 배치 사이즈가 1이므로 배치 차원을 없앤다.
    conv_output = conv_output[0]
    grad_val = grad_val[0]

    ## 구한 gradient를 픽셀 가로세로로 평균내서 a^c_k를 구한다.
    weights = np.mean(grad_val, axis=(0, 1))

    ## 추출한 conv_output에 weight를 곱하고 합하여 grad_cam을 얻는다.
    grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for k, w in enumerate(weights):
        grad_cam += w * conv_output[:, :, k]

    grad_cam_f = cv2.resize(grad_cam, (32, 32))

    ## ReLU를 씌워 음수를 0으로 만든다.
    grad_cam_f = np.maximum(grad_cam, 0)

    grad_cam_f = grad_cam_f / grad_cam_f.max()
    return grad_cam_f, img_arr


def normalize_production(x):
    # this function is used to normalize instances in production according to saved training set statistics
    # Input: X - a training set
    # Output X - a normalized training set according to normalization constants.

    # these values produced during first training and are general for the standard cifar10 training set normalization
    # for transfer
    mean = 120.06
    std = 63.41
    return (x - mean) / (std + 1e-7)
# (img_tensor, model, class_index, activation_layer):
def generate_grad_cam(img_tensor, model, class_idx, activation_layer):
    ## img_path -> preprocessed image tensor
    # img_arr, img_tensor = preprocess_input(img_path)
    input_tensor = np.expand_dims(img_tensor, 0)
    input_tensor = normalize_production(input_tensor)

    img_arr = np.asarray(img_tensor) / 255.
    # img_tensor = np.expand_dims(img_arr, 0)
    ## get the derivative of y^c w.r.t A^k
    y_c = model.layers[-1].output.op.inputs[0][0, class_idx]
    #     y_c = model.output[0, class_idx]

    #     layer_output = model.get_layer('block5_conv3').output
    # layer_output = model.get_layer[activation_layer].output
    layer_output = model.get_layer(activation_layer).output
    grads = K.gradients(y_c, layer_output)[0]
    gradient_fn = K.function([model.input], [layer_output, grads, model.layers[-1].output])

    conv_output, grad_val, predictions = gradient_fn([input_tensor])
    conv_output, grad_val = conv_output[0], grad_val[0]

    weights = np.mean(grad_val, axis=(0, 1))
    cam = np.dot(conv_output, weights)

    cam = cv2.resize(cam, (32, 32))

    ## Relu
    cam = np.maximum(cam, 0)

    cam = cam / cam.max()
    return cam, img_arr  # , predictions


def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'daughter']


def show_filter(model, img_tensor, layer_list ):
    for layer_name in layer_list:
        activation_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        input_tensor = np.expand_dims(img_tensor, 0)
        layer_activation = activation_model.predict(input_tensor)

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


def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_integer('training_epochs', 15, '# of step for training')
    flags.DEFINE_integer('test_interval', 1000, '# of interval to test_img a model')
    flags.DEFINE_integer('summary_interval', 100, '# of step to save summary')
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    flags.DEFINE_float('gpu', 0, 'gpu use :1, cpu :0')
    # flags.DEFINE_string('training_type', 'transfer', 'Choose one [all, transfer]')
    flags.DEFINE_integer('run_type', 4, 'transfer_predict : 0, train : 1, transfer_learning : 2,  show_cam : 3, show_filter :4')
    flags.DEFINE_string('additional_img_path', './preprocess/augmentation/', 'addtional images')

    # data

    # Run_Type
    # flags.DEFINE_integer('run_type', 1, '1 : train, 2 : predict')
    # fix bug of flags
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


if __name__ == '__main__':
    conf = configure()
    tfconfig = tf.ConfigProto()
    if conf.gpu == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.3
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from keras.models import Model

    run_type = conf.run_type
    if run_type == 0:

        fig, axes = plt.subplots(2, 3, figsize=(8, 6))
        fig.suptitle("Trainsfer Learning Result(Adding daughter class)")
        #
        model = cifar10vgg(run_type)
        # Additional class daughter,
        predict_img1 = cv2.imread("./preprocess/test_img/3.png")
        predict_img2 = cv2.imread("./preprocess/test_img/1.png")
        predict_img3 = cv2.imread("./preprocess/test_img/2.png")

        convert_img1 = predict_img1[..., ::-1].copy()
        convert_img2 = predict_img2[..., ::-1].copy()
        convert_img3 = predict_img3[..., ::-1].copy()

        arr_img1 = predict_img3 / 255.
        axes[0, 0].imshow(convert_img1 / 255)
        axes[0, 1].imshow(convert_img2 / 255)
        axes[0, 2].imshow(convert_img3 / 255)


        input_tensor1 = np.expand_dims(convert_img1, 0)
        input_tensor2 = np.expand_dims(convert_img2, 0)
        input_tensor3 = np.expand_dims(convert_img3, 0)

        result1 = model.predict(input_tensor1)
        result2 = model.predict(input_tensor2)
        result3 = model.predict(input_tensor3)

        axes[0, 0].set_title(load_label_names()[np.argmax(result1)])
        axes[0, 1].set_title(load_label_names()[np.argmax(result2)])
        axes[0, 2].set_title(load_label_names()[np.argmax(result3)])

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x1 = random.randint(0, 10000)
        x2 = random.randint(0, 10000)
        x3 = random.randint(0, 10000)

        predict_img4 = x_test[x1].copy()
        predict_img5 = x_test[x2].copy()
        predict_img6 = x_test[x3].copy()

        predict_img4_label = y_train[x1]
        predict_img5_label = y_train[x2]
        predict_img6_label = y_train[x3]

        convert_img4 = predict_img4[..., ::-1].copy()
        convert_img5 = predict_img5[..., ::-1].copy()
        convert_img6 = predict_img6[..., ::-1].copy()

        input_tensor4 = np.expand_dims(convert_img4, 0)
        input_tensor5 = np.expand_dims(convert_img5, 0)
        input_tensor6 = np.expand_dims(convert_img6, 0)

        result4 = model.predict(input_tensor4)
        result5 = model.predict(input_tensor5)
        result6 = model.predict(input_tensor6)

        axes[1, 0].imshow(convert_img4 / 255)
        axes[1, 1].imshow(convert_img5 / 255)
        axes[1, 2].imshow(convert_img6 / 255)
        axes[1, 0].set_title(load_label_names()[np.argmax(result4)])
        axes[1, 1].set_title(load_label_names()[np.argmax(result5)])
        axes[1, 2].set_title(load_label_names()[np.argmax(result6)])
        plt.show()



    elif run_type in (1, 2):
        (x_train, y_train), (x_test_ori, y_test_ori) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test_ori.astype('float32')

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test_ori, 10)

        model = cifar10vgg(run_type)

    elif run_type == 3:
        model = cifar10vgg(run_type)
        # Additional class daughter,
        #fig, axes = plt.subplots(10, 20, figsize=(2, 2))
        plot_width = 20
        plot_height = 3
        fig, axes = plt.subplots(plot_height*3, plot_width)
        plt.gca().axes.get_yaxis().set_visible(False)
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_test = x_test.astype('float32')
        y_test = y_test.astype('float32')
        cam_x = x_test[0]
        cam_y = int(y_test[0])
        print(load_label_names()[cam_y])
        #predicted_x = model.predict(cam_x)
        # cam, img = generate_grad_cam_test(cam_x, model.model, cam_y,'conv2d_13')
        layer_name = "activation_13"
        layer_name  = ['activation_13', 'activation_8', 'activation_4']
        for width_plot in range(plot_width):
            for height_plot in range(plot_height):
                x = random.randint(0, 9999)
                cam_x = x_test[x]
                cam_y = int(y_test[x])
                cam, img = generate_grad_cam(cam_x, model.model, cam_y, layer_name[height_plot])
                row_number = height_plot*3
                axes[row_number, width_plot].imshow(img)
                axes[row_number+1, width_plot].imshow(cam)
                axes[row_number+2, width_plot].imshow(img)
                axes[row_number+2, width_plot].imshow(cam, cmap='jet', alpha=0.5)
                axes[row_number, width_plot].set_title(load_label_names()[cam_y])
                axes[row_number, width_plot].axis('off')
                axes[row_number+1, width_plot].axis('off')
                axes[row_number+2, width_plot].axis('off')

        plt.tight_layout()
        plt.show()

    elif run_type == 4:
        model = cifar10vgg(run_type)
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_test = x_test.astype('float32')
        y_test = y_test.astype('float32')
        x = random.randint(0, 9999)
        cam_x = x_test[x]
        cam_y = int(y_test[x])
        layer_list = ["conv2d_1","conv2d_10"]
        cam, img = show_filter(model.model, cam_x, layer_list)
