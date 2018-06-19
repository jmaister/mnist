# https://keras.io/layers/core/
# https://keras.io/layers/convolutional/

import argparse

#[Import dependencies]
import numpy as np

from sklearn.preprocessing import *

import keras
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2

from ConvTrainer import ConvTrainer
from ConvTrainerSimpler import ConvTrainerSimpler

import mnist

data_folder = '/data/'
output_folder = '/output/'

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

epochs = 20

num_classes = 10
img_rows = 28
img_cols = 28
input_shape = (28, 28)

def main():
    global data_folder
    global output_folder

    parser = argparse.ArgumentParser(description='Add features.')
    parser.add_argument('-l', dest='local', action='store_true', help='Use local folders', default=False)

    args = parser.parse_args()
    print(args.local)

    if args.local:
        data_folder = './data/'
        output_folder = './output/'
    print('data', data_folder)
    print('output', output_folder)

    train()

def train():
    global data_folder
    global output_folder

    #load data
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    # X, Y
    X = train_images
    Y = keras.utils.to_categorical(train_labels, num_classes=num_classes)
    X_test = test_images
    Y_test = keras.utils.to_categorical(test_labels, num_classes=num_classes)

    print('Image data format', K.image_data_format())

    if K.image_data_format() == 'channels_first':
        X = X.reshape(X.shape[0], 1, img_rows, img_cols)
        X_test = test_images.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X = X.reshape(X.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    trainer = ConvTrainer(input_shape)


    # X, Y = trainer.prepare_data(X, Y)
    print('X shape', X.shape)
    print('Y shape', Y.shape)

    model = trainer.create_model()
    model.summary()

    model.fit(X, Y, validation_data=(X_test, Y_test), shuffle=True, epochs=epochs, batch_size=500)
    print('* Done training')

    # serialize model to JSON
    print("* Saving model to disk..")
    model_json = model.to_json()
    with open(output_folder + "mnist_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(output_folder + "mnist_model.h5")
    print("* Saved model  to disk")

    #Compute accuracy scores
    print("* Computing accuracy scores..")
    score = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[0], score[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

if __name__ == '__main__':
    main()
