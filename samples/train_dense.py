# https://keras.io/layers/core/
# https://keras.io/layers/convolutional/

#[Import dependencies]
import numpy as np
import pandas

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

epochs = 20

def create_model():
    model = Sequential()
    #kernel_initializer='normal'
    model.add(Dense(64, input_dim=50, activation='relu', W_regularizer=l2(0.001)))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu', W_regularizer=l2(0.001) ))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu', W_regularizer=l2(0.001) ))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu', W_regularizer=l2(0.001) ))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu', W_regularizer=l2(0.001) ))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='relu', W_regularizer=l2(0.001) ))

    print('Modeled Network')
    model.summary()

    # Compile model
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model

def main():

    #load data
    dataframe = pandas.read_csv("/data/numerai_training_data.csv")
    dataset = dataframe.values

    # split into input (X) and output (Y) variables
    X = dataset[:,3:53].astype(float) #X f(eatures) are from the first column and the 50th column
    Y = dataset[:,53] # Y (lables) are from the 50th column

    # regularize
    X = (X - np.min(X)) / np.ptp(X)
    
    print('X shape', X.shape)
    print('Y shape', Y.shape)

    # reshape
    #X = np.expand_dims(X, axis=2)
    print('X shape', X.shape)
    print('Y shape', Y.shape)

    model = create_model()
    
    model.fit(X, Y, validation_split=0.2, shuffle=True, epochs=epochs, batch_size=1000)
    print('* Done training')

    # serialize model to JSON
    print("* Saving model to disk..")
    model_json = model.to_json()
    with open("/output/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("/output/model.h5")
    print("* Saved model  to disk")

    #Compute accuracy scores
    print("* Computing accuracy scores..")
    score = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[0], score[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

if __name__ == '__main__':
    main()
