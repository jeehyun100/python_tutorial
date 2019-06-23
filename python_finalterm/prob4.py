# -*- coding: utf-8 -*-
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.optimizers import Adam
from matplotlib import pyplot as plt

"""
딥러닝 기본 훈련 변수
    batch_size, num_classes, epochs
"""
batch_size = 200
num_classes = 10
epochs = 20


def preprocess_mnist_data():
    """
    Mnist 데이터를 Processing 하는 기능

    Return:
        x_train, numpy.array : X input data for training
        y_train, numpy.array : Y input data for training
        x_test, numpy.array : X input data for testing
        y_test, numpy.array : Y input data for testing
    """
    (_x_train, _y_train), (_x_test, _y_test) = mnist.load_data()

    _x_train = _x_train.reshape(60000, 784)
    _x_test = _x_test.reshape(10000, 784)
    _x_train = _x_train.astype('float32')
    _x_test = _x_test.astype('float32')
    # 255로 나누어서 0~1 사이의 float형으로 normalize 한다.
    _x_train /= 255
    _x_test /= 255

    # label을 onehot vector 방식으로 전처리 한다.
    _y_train = keras.utils.to_categorical(_y_train, num_classes)
    _y_test = keras.utils.to_categorical(_y_test, num_classes)
    return _x_train, _y_train, _x_test, _y_test


def base_model(nn, opti):
    """
    hidden layer 수와, Optimizer로 DNN model을 만드는 기능
        * weight initial은 Xavier(glorot_normal)을 사용
        * learning rate는 0.001
        * activation은 relu

    Args:
        nn, int : Hidden layer number of model
        opti, String : The type of optimizer
    Return:
        model, Keras.models: keras.model
    """
    model = Sequential()
    model.add(Dense(nn, kernel_initializer='glorot_normal', activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(nn, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nn, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, kernel_initializer='glorot_normal', activation='softmax'))

    model.summary()
    _optimizer = SGD(lr=0.001)

    if opti == 'Adam':
        _optimizer = Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=_optimizer,
                  metrics=['accuracy'])
    return model


def model_train_eval(hidden_layer, optimizer, plot_losses, x_tr, y_tr, x_ts, y_ts):
    """
    hidden layer 수와, Optimizer와, Metric 저장용 callback 함수를
    인자로 받아 모델 생성 후 train과 evaluation을 하는 기능

    Args:
        hidden_layer, int : Hidden layer number of model
        optimizer, String : The type of optimizer
        plot_losses, keras.callbacks.Callback : Save metric callback class
        x_tr, numpy.array : X input data for training
        y_tr, numpy.array : Y input data for training
        x_ts, numpy.array : X input data for testing
        y_ts, numpy.array : Y input data for testing
    Return:
        history, Keras.callbacks.History : The list of val_loss,val_acc, loss, acc
        score, list : The list of scalar metric
    """
    fc_model = base_model(hidden_layer, optimizer)
    history = fc_model.fit(x_tr, y_tr,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           callbacks=[plot_losses],
                           validation_data=(x_ts, y_ts))
    score = fc_model.evaluate(x_ts, y_ts, verbose=0)
    return history, score


def plot_losses_acc_helper(axis, model_type, plot_losses, optimizer):
    """
    문제 4번에 맞게 train curve와 추가로 loss curve를 Plotting 하는 기능

    Args:
        axis, matplotlib.axis tuple : matplot sybplot axis
        model_type, int : Hidden layer number of model
        plot_losses, keras.callbacks.Callback : Save metric callback class
        optimizer, String : The type of optimizer

    """
    _ax1 = axis[0]
    _ax2 = axis[1]
    _ax1.title.set_text("Loss curve " + optimizer)
    _ax1.set_yscale('log')
    train_loss_label = "{0}{1}".format(model_type, "loss")
    val_loss_label = "{0}{1}".format(model_type, "val_loss")
    train_acc_label = "{0}{1}".format(model_type, "accuracy")
    val_acc_label = "{0}{1}".format(model_type, "val_accuracy")

    _ax1.plot(plot_losses.x, plot_losses.losses, label=train_loss_label)
    _ax1.plot(plot_losses_64.x, plot_losses.val_losses, label=val_loss_label)

    _ax1.legend()
    _ax2.title.set_text("Train curve " + optimizer)
    _ax2.plot(plot_losses.x, plot_losses.acc, label=train_acc_label)
    _ax2.plot(plot_losses.x, plot_losses.val_acc, label=val_acc_label)
    _ax2.legend()


class PlotLosses(keras.callbacks.Callback):
    """
    실시간 plotting을 위한 callback 함수
    epoch 끝날때마다 정해진 memtric을 저장한다.
     * on_epoch_end 안에 plotting 메소드를 넣으면 매 epoch마다 plotting을 할수있다.
     * jupyter에서는 매 epoch마다 동적으로 그릴수 있으나, pycharm에서는 안된다.
    """

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1


if __name__ == "__main__":
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(10, 10))

    x_train, y_train, x_test, y_test = preprocess_mnist_data()
    # 실시간으로 plotting을 할려고 별도의 callback함수를 만들었으나 jupyter에서만 동작

    # 문제 4.(가) 주어진 대로 네트워크를 만들고 train curve를 그리시오
    plot_losses_64 = PlotLosses()
    history64, score64 = model_train_eval(64, 'SGD', plot_losses_64, x_train, y_train, x_test, y_test)
    plot_losses_acc_helper((ax1, ax2), 64, plot_losses_64, 'SGD')

    # 문제 4.(나) 노드를 512로 변환하고 네트워크를 만들고 train curve를 그리시오
    plot_losses_512 = PlotLosses()
    history512, score512 = model_train_eval(512, 'SGD', plot_losses_512, x_train, y_train, x_test, y_test)
    plot_losses_acc_helper((ax1, ax2), 512, plot_losses_512, 'SGD')

    # 문제 4.(다) Optimizer를 Adam으로 바꾸고, 결과를 비교하시오.
    # Adam이 SGD 보다 훨신 빠른속도로 훈련이 된다.
    # node 512인 경우 train data에 Overfitting 되는 경향을 보인다.
    plot_losses_adam_64 = PlotLosses()
    history64_adam, score64_adam = model_train_eval(64, 'Adam', plot_losses_adam_64, x_train, y_train, x_test, y_test)
    plot_losses_acc_helper((ax3, ax4), 64, plot_losses_adam_64, 'Adam')

    plot_losses_adam_512 = PlotLosses()
    history512_adam, score512_adam = model_train_eval(512, 'Adam',
                                                      plot_losses_adam_64, x_train, y_train, x_test, y_test)
    plot_losses_acc_helper((ax3, ax4), 512, plot_losses_adam_64, 'Adam')

    plt.show()
