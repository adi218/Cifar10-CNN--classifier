import keras
import numpy as np
from keras.datasets import cifar10

class Preprocessor:
    labels_train_arr = []
    labels_test = None

    def load_data(self):

        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        np.moveaxis(X_train, 0, -1)
        np.moveaxis(X_test, 0, -1)
        X_train = X_train / 255.
        X_test = X_test / 255
        return X_train, Y_train, X_test, Y_test


