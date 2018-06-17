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

#load data
dataframe = pandas.read_csv("data/numerai_training_data.csv")
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,3:53].astype(float) #X f(eatures) are from the first column and the 50th column
Y = dataset[:,53] # Y (labels) are from the 50th column

print('X shape', X.shape)
print('Y shape', Y.shape)

# reshape
X = np.expand_dims(X, axis=2)

#Example Neural Network Architecture
#Define Neural network architecture of 10 Hidden layer with 500 Neurons each
model = Sequential()
#model.add(Dense(128, input_dim=50, kernel_initializer='normal', activation='relu'))
#model.add(Dropout(0.25))
#model.add(Dense(128, activation='relu', W_regularizer=l2(0.001) ))

model.add(Conv1D(128, 3, strides=1, padding='same', activation='relu', W_regularizer=l2(0.001), kernel_initializer='glorot_uniform', input_shape=(50, 1)))
model.add(Conv1D(256, 3, strides=1, padding='same', activation='relu', W_regularizer=l2(0.001), kernel_initializer='glorot_uniform'))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Conv1D(256, 3, strides=1, padding='same', activation='relu', W_regularizer=l2(0.001), kernel_initializer='glorot_uniform'))
model.add(MaxPooling1D(pool_size=(2)))

# Back to NN
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='glorot_uniform'))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='glorot_uniform'))
model.add(Dense(1, activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='glorot_uniform'))
model.summary()
print('Modeled Network')

# Compile model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print('* Finished compiling model')

model.fit(X, Y, validation_split=0.2, shuffle=True, epochs=20, batch_size=4000)
print('* Done training')

# serialize model to JSON
print("* Saving model to disk..")
model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/model.h5")
print("* Saved model  to disk")

#Compute accuracy scores
print("* Computing accuracy scores..")
score = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
print(model.metrics_names)

