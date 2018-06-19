import numpy as np
import pandas

from sklearn.preprocessing import *

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2

class ConvTrainer:

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def get_name(self):
        return 'conv1d_model'
    
    def prepare_data(self, X, Y):
        return X, Y
    
    def create_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=self.input_shape))
        model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))

        # Back to NN
        model.add(Flatten())
        model.add(Dense(32, activation='sigmoid'))
        model.add(Dense(10, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model
