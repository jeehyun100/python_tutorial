'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.optimizers import Adam
from matplotlib import pyplot as plt
#from IPython.display import clear_output



batch_size = 200
num_classes = 10
epochs = 50

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# updatable plot
# a minimal example (sort of)

class PlotLosses(keras.callbacks.Callback):
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




def base_model(nn, opti):
    model = Sequential()
    model.add(Dense(nn, kernel_initializer='glorot_normal',activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(nn, kernel_initializer='glorot_normal' ,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nn, kernel_initializer='glorot_normal' ,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes,kernel_initializer='glorot_normal', activation='softmax'))

    model.summary()
    if opti == 'SGD':
        _optimizer = SGD(lr=0.001)
    elif opti == 'Adam':
        _optimizer = Adam(lr=0.001)


    model.compile(loss='categorical_crossentropy',
                  optimizer=_optimizer,
                  metrics=['accuracy'])
    return model

fc_model64 = base_model(64, 'SGD')
plot_losses = PlotLosses()
history64 = fc_model64.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[plot_losses],
                    validation_data=(x_test, y_test))
score64 = fc_model64.evaluate(x_test, y_test, verbose=0)


fc_model512 = base_model(512, 'SGD')
plot_losses512 = PlotLosses()
history512 = fc_model512.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[plot_losses512],
                    validation_data=(x_test, y_test))
score512 = fc_model512.evaluate(x_test, y_test, verbose=0)


fc_model64_adam = base_model(64, 'Adam')
plot_losses_adam = PlotLosses()
history64_adam = fc_model64_adam.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[plot_losses_adam],
                    validation_data=(x_test, y_test))
score64_adam = fc_model64_adam.evaluate(x_test, y_test, verbose=0)


fc_model512_adam = base_model(512, 'Adam')
plot_losses512_adam = PlotLosses()
history512_adam = fc_model512_adam.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[plot_losses512_adam],
                    validation_data=(x_test, y_test))
score512_adam = fc_model512_adam.evaluate(x_test, y_test, verbose=0)

#self.fig = plt.figure()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True,figsize=(10,10))
ax1.title.set_text('SGD')
ax1.set_yscale('log')
ax1.plot(plot_losses.x, plot_losses.losses, label="loss")
ax1.plot(plot_losses.x, plot_losses.val_losses, label="val_loss")
ax1.plot(plot_losses512.x, plot_losses512.losses, label="512loss")
ax1.plot(plot_losses512.x, plot_losses512.val_losses, label="512val_loss")
ax1.legend()

ax2.plot(plot_losses.x, plot_losses.acc, label="accuracy")
ax2.plot(plot_losses.x, plot_losses.val_acc, label="validation accuracy")
ax2.plot(plot_losses512.x, plot_losses512.acc, label="512accuracy")
ax2.plot(plot_losses512.x, plot_losses512.val_acc, label="512validation accuracy")
ax2.legend()

# ax3.title.set_text('512,SGD')
# ax3.set_yscale('log')
# ax3.plot(plot_losses512.x, plot_losses512.losses, label="512loss")
# ax3.plot(plot_losses512.x, plot_losses512.val_losses, label="512val_loss")
# ax3.legend()
#
# ax4.plot(plot_losses512.x, plot_losses512.acc, label="512accuracy")
# ax4.plot(plot_losses512.x, plot_losses512.val_acc, label="512validation accuracy")
# ax4.legend()

ax3.title.set_text('Adam')
ax3.set_yscale('log')
ax3.plot(plot_losses_adam.x, plot_losses512.losses, label="loss")
ax3.plot(plot_losses_adam.x, plot_losses512.val_losses, label="val_loss")
ax3.plot(plot_losses512_adam.x, plot_losses512_adam.losses, label="512_loss")
ax3.plot(plot_losses512_adam.x, plot_losses512_adam.val_losses, label="512_val_loss")
ax3.legend()

ax4.plot(plot_losses_adam.x, plot_losses512.acc, label="accuracy")
ax4.plot(plot_losses_adam.x, plot_losses512.val_acc, label="validation accuracy")
ax4.plot(plot_losses512_adam.x, plot_losses512_adam.acc, label="512_accuracy")
ax4.plot(plot_losses512_adam.x, plot_losses512_adam.val_acc, label="512_validation accuracy")
ax4.legend()

# ax7.title.set_text('512,Adam')
# ax7.set_yscale('log')
# ax7.plot(plot_losses512_adam.x, plot_losses512_adam.losses, label="loss")
# ax7.plot(plot_losses512_adam.x, plot_losses512_adam.val_losses, label="val_loss")
# ax7.legend()
#
# ax8.plot(plot_losses512_adam.x, plot_losses512_adam.acc, label="accuracy")
# ax8.plot(plot_losses512_adam.x, plot_losses512_adam.val_acc, label="validation accuracy")
# ax8.legend()

plt.show()

print('Test loss:', score[0])
print('Test accuracy:', score[1])