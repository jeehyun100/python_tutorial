
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

class cifar10vgg:
    def __init__(self,train=True):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.33
        set_session(tf.Session(config=config))

        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cifar10vgg.h5')


    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model):

        #training parameters
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        #data augmentation
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



        #optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


        # training process in a for loop with learning rate drop every 25 epoches.

        historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)
        model.save_weights('cifar10vgg.h5')
        return model


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

    img_arr = np.asarray(img_tensor)[0,:, :, :3] / 255.
    #img_arr = np.asarray(img_tensor)[0, :, :, :3]
    #img_tensor = img_arr#np.expand_dims(img_arr, 0)

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

#(img_tensor, model, class_index, activation_layer):
def generate_grad_cam(img_tensor, model, class_idx, activation_layer):
    ## img_path -> preprocessed image tensor
    #img_arr, img_tensor = preprocess_input(img_path)
    img_arr = np.asarray(img_tensor)[0,:, :, :3] / 255.
    #img_tensor = np.expand_dims(img_arr, 0)
    ## get the derivative of y^c w.r.t A^k
    y_c = model.layers[-1].output.op.inputs[0][0, class_idx]
    #     y_c = model.output[0, class_idx]

    #     layer_output = model.get_layer('block5_conv3').output
    #layer_output = model.get_layer[activation_layer].output
    layer_output = model.get_layer(activation_layer).output
    grads = K.gradients(y_c, layer_output)[0]
    gradient_fn = K.function([model.input], [layer_output, grads, model.layers[-1].output])

    conv_output, grad_val, predictions = gradient_fn([img_tensor])
    conv_output, grad_val = conv_output[0], grad_val[0]

    weights = np.mean(grad_val, axis=(0, 1))
    cam = np.dot(conv_output, weights)

    cam = cv2.resize(cam, (32, 32))

    ## Relu
    cam = np.maximum(cam, 0)

    cam = cam / cam.max()
    return cam, img_arr#, predictions

def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def show_filter(vgg):
    from tensorflow.python.keras import models

    layer_outputs = [layer.output for layer in vgg.model.layers[:11]]
    activation_model = models.Model(inputs=vgg.model.input, outputs=layer_outputs)

    # 층의 이름을 그래프 제목으로
    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)

    images_per_row = 16

    # 여기서부터가 특징맵 출력하는 부분입니다.
    for layer_name, layer_activation in zip(layer_names, activations):
        # 특징의 수를 뜻합니다.
        # 정확히 각 특징의 채널의 수라고 표현하면 이해가 쉽겠죠?
        n_features = layer_activation.shape[-1]

        # 특징맵의 크기는 (1, size, size, n_features)입니다., 1은 batch_size
        # layer_activation.shape[1]이 이미지의 width, height거든요.
        size = layer_activation.shape[1]

        # 활성화를 보여주기위한 grid를 정의합니다.
        n_cols = n_features // images_per_row  # 열의 갯수를 16개의 단위로 보여주고 싶어서입니다.
        # 첫번째 층은 32개니 2개의 row로 나타나겟죠?

        # 이미지가 들어가야할 픽셀 갯수라고 생각하시면 됩니다.
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # 각 활성화를 채우는 부분.
        # 특성맵 갯수에 따른 행의 길이 : n_cols
        for col in range(n_cols):
            # 16개씩 채우면서.
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

if __name__ == '__main__':
    from keras.models import Model

    fig, axes = plt.subplots(4, 5, figsize=(8, 6))
    (x_train, y_train), (x_test_ori, y_test_ori) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test_ori.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test_ori, 10)

    model = cifar10vgg(False)

    cam_x = x_test[0]
    cam_y = int(y_test_ori[0])
    print (load_label_names()[cam_y] )
    cam_x = np.expand_dims(cam_x, 0)
    predicted_x = model.predict(cam_x)
    #cam, img = generate_grad_cam_test(cam_x, model.model, cam_y,'conv2d_13')
    layer_name = "conv2d_8"
    cam, img = generate_grad_cam(cam_x, model.model, cam_y, layer_name)
    axes[0, 0].imshow(img)
    axes[1, 0].imshow(cam)
    axes[2, 0].imshow(img)
    axes[2, 0].imshow(cam, cmap='jet', alpha=0.5)
    plt.tight_layout()
    plt.show()



    layer_activation = model.model.get_layer(layer_name)
    layer_activation = layer_activation.output
    activation_model = Model(inputs=model.model.input, outputs=model.model.get_layer(layer_name).output)
    activations = activation_model.predict(cam_x)
    layer_activation = activations
    # # layer_names = []
    # # for layer in model.layers[:8]:
    # #     layer_names.append(layer.name)
    #
    # plt.matshow(layer_activation[0, :, :, 19], cmap='viridis')
    #
    # plt.show()



    images_per_row = 16

    # # 여기서부터가 특징맵 출력하는 부분입니다.
    # for layer_name, layer_activation in zip(layer_names, activations):
    #     # 특징의 수를 뜻합니다.
    #     # 정확히 각 특징의 채널의 수라고 표현하면 이해가 쉽겠죠?
    n_features = layer_activation.shape[-1]

    # 특징맵의 크기는 (1, size, size, n_features)입니다., 1은 batch_size
    # layer_activation.shape[1]이 이미지의 width, height거든요.
    size = layer_activation.shape[1]

    # 활성화를 보여주기위한 grid를 정의합니다.
    n_cols = n_features // images_per_row  # 열의 갯수를 16개의 단위로 보여주고 싶어서입니다.
    # 첫번째 층은 32개니 2개의 row로 나타나겟죠?

    # 이미지가 들어가야할 픽셀 갯수라고 생각하시면 됩니다.
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # 각 활성화를 채우는 부분.
    # 특성맵 갯수에 따른 행의 길이 : n_cols
    for col in range(n_cols):
        # 16개씩 채우면서.
        for row in range(images_per_row):
            # # 모든 채널을 사용하겠다.
            # channel_image = layer_activation[0,
            #                 :, :,
            #                 col * images_per_row + row]
            # 픽셀로 표현하기 위해.
            # 모든 채널을 사용하겠다.
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]
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





    # fig = plt.figure(figsize=(8, 8))
    # plt.axis('off')
    #first_layer_activation.output[0, :, :, 19]

    #plt.matshow(first_layer_activation.output[0, :, :, 19])

    # plt.tight_layout()
    # plt.show()
    #show_filter(model)

    # print(load_label_names()[cam_y])
    # #img, cam, predictions = generate_grad_cam(model.model, cam_x, cam_y)
    # #img_tensor, model, class_index, activation_layer)
    # cam, img = generate_grad_cam_test(cam_x, model.model, cam_y,'conv2d_13')
    #
    # axes[0, 0].imshow(img)
    # axes[1, 0].imshow(cam)
    # axes[2, 0].imshow(img)
    # axes[2, 0].imshow(cam, cmap='jet', alpha=0.5)
    # plt.tight_layout()
    # plt.show()

    # predicted_x = model.predict(x_test)
    # residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)
    #
    # loss = sum(residuals)/len(residuals)
    # print("the validation 0/1 loss is: ",loss)



