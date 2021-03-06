import numpy as np
import pandas

from sklearn.preprocessing import *

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2

class ConvTrainerSimpler:

    def __init__(self):
        pass

    def get_name(self):
        return 'ConvTrainerSimpler'
    
    def prepare_data(self, X, Y):
        return X, Y
    
    def create_model(self):
        model = Sequential()

        model.add(Conv1D(10, 3, strides=1, padding='same', activation='relu', input_shape=(28, 28)))
        model.add(Conv1D(10, 3, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=(2)))
        model.add(Conv1D(10, 3, strides=1, padding='same', activation='relu'))

        # Back to NN
        model.add(Flatten())
        model.add(Dense(10, activation='sigmoid'))
        model.add(Dense(10, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model
