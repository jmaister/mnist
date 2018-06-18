import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

from sklearn.preprocessing import MinMaxScaler

import numpy as np
import os


#load data
import cv2


X = np.empty((10, 28, 28))
for i in range(0,10):
    print(i)
    input_image = cv2.imread("./data/test-"+ str(i) +".png", cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(input_image, (28, 28))
    grey_values = 255 - resized_image
    X[i] = grey_values

print('X', X.shape)

#reshape
#X = np.expand_dims(X, axis=2)

# load json and create model
json_file = open('./output/mnist_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("./output/mnist_model.h5")
print("Loaded model from disk")
 
predictions = loaded_model.predict(X)

print('pred', predictions.shape)
print(predictions)
for p in predictions:
    print(np.argmax(p))
