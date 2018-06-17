# https://keras.io/layers/core/
# https://keras.io/layers/convolutional/

#[Import dependencies]
import sys, getopt
import argparse

import numpy as np
import pandas as pd

from sklearn.preprocessing import *

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

epochs = 20

data_folder = '/data/'
output_folder = '/output/'
chunk_size = 3000
chunks = 999999999 
features = ['feature' + str(x+1) for x in range(50)]


def add_features(X):
    print('adding features')
    newfeatures = []
    for n in range(50):
        for m in range(50):
            newname = 'new_' + str(n) + 'x' + str(m)
            X.loc[:, (newname)] = X.loc[:,(features[n])] * X.loc[:, (features[m])]
            newfeatures.append(newname)

    print('features added')
    return (X, newfeatures)

def main():
    global data_folder
    global output_folder
    global chunk_size
    global chunks
    global features

    parser = argparse.ArgumentParser(description='Add features.')
    parser.add_argument('-l', dest='local', action='store_true', help='Use local folders', default=False)

    args = parser.parse_args()
    print(args.local)

    if args.local:
        data_folder = 'c:/workspace/numeraifloyddata/'
        output_folder = 'c:/workspace/numeraifloydresults/'
        chunk_size = 10
        chunks = 1
    print('data', data_folder)
    print('output', output_folder)
    
    #load data
    #dataframe = pd.read_csv("/data/numerai_training_data.csv")
    dataframe = pd.read_csv(data_folder + "numerai_training_data.csv", iterator=True, chunksize=chunk_size)
    #print(dataframe.head())
    #print(dataframe.info(memory_usage='deep'))
    
    '''
    # split into input (X) and output (Y) variables
    X = dataset[:,3:53].astype(float) #X f(eatures) are from the first column and the 50th column
    Y = dataset[:,53] # Y (lables) are from the 50th column

    print('X shape', X.shape)
    print('Y shape', Y.shape)
    '''

    clf = RandomForestClassifier(n_jobs=-1, n_estimators=20)

    chunk_features = None

    c = 0
    for chunk in dataframe:

        if c > chunks:
            break
        else:
            c = c + 1

        # create new features
        X, new_features = add_features(chunk)
        if not chunk_features:
            chunk_features = features + new_features
        print('X shape', X.shape)
        print('X new column', (X[features[1]] * X[features[2]]).shape)
        print('new features', len(chunk_features))
        Y = chunk['target']

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, random_state=0)

        clf.fit(X_train[chunk_features], y_train)
        score = clf.score(X_test[chunk_features], y_test)
        print('score %f' % score)
    
    # results
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)

    print(importances)
    print(sorted_idx)

    result = np.array((2,1))

    c = 0

    dataframe_predict = pd.read_csv(data_folder + "numerai_tournament_data.csv", iterator=True, chunksize=chunk_size)
    for chunk in dataframe_predict:

        if c > chunks:
            break
        else:
            c = c + 1

        X_predict, _ = add_features(chunk)
        predictions = clf.predict_proba(X_predict[chunk_features])
    
        print('id shape', X_predict['id'].shape)
        ids = X_predict['id'].values.reshape((-1, 1))
        predictions = predictions[:,1].reshape(predictions.shape[0], 1)
        print('d', ids.shape)
        print('p', predictions.shape)

        partial_result = np.concatenate((ids, predictions), axis=1)
    
        result = np.vstack((result, partial_result))

    print('r', result.shape)

    prediction_file = output_folder + 'forest_prediction_01.csv'
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

    for i in sorted_idx:
        print(chunk_features[i])

    '''
    padding = np.arange(len(features)) + 0.5
    plt.barh(padding, importances[sorted_idx], align='center')
    plt.yticks(padding, features[sorted_idx])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")
    plt.show()
    '''

    

if __name__ == '__main__':
    main()
