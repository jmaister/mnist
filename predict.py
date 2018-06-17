from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

from sklearn.preprocessing import MinMaxScaler

import numpy as np
import os
import pandas

from train import normalize

prediction_file = '/output/predictions_01.csv'

#load data
dataframe = pandas.read_csv("/data/numerai_tournament_data.csv")
dataset = dataframe.values
X = dataset[:,3:53]

#X = normalize(X)

#reshape
X = np.expand_dims(X, axis=2)

# load json and create model
json_file = open('/model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("/model/model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print "%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100)

print('* Making predictions..')
predictions = loaded_model.predict(X)

# preditions must be between 0.3 and 0.7
#scaler = MinMaxScaler(feature_range=(0.3001, 0.6999))
#scaler = scaler.fit(predictions)
#predictions = scaler.transform(predictions)

ids = np.reshape(dataset[:,0], (dataset[:,0].shape[0], 1))
print('d', ids.shape)
print('p', predictions.shape)

result = np.concatenate((ids, predictions), axis=1)

print('r', result.shape)

np.savetxt(
    prediction_file,          # file name
    result,  # array to save
    fmt='%s,%.9f',               # formatting, 2 digits in this case
    delimiter=',',          # column delimiter
    newline='\n',           # new line character
    header= 'id,probability',   # file header
	comments= ''             # avoid '#' in the header
)

print('Predictions saved to ', prediction_file)
