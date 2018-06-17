# https://keras.io/layers/core/
# https://keras.io/layers/convolutional/

#[Import dependencies]
import numpy as np
import pandas

from sklearn.preprocessing import *

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2

from ConvTrainer import ConvTrainer
from DenseTrainerV1 import DenseTrainer

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

epochs = 100


def main():
    trainer = DenseTrainer()

    #load data
    dataframe = pandas.read_csv("/data/numerai_training_data.csv")
    dataset = dataframe.values

    # split into input (X) and output (Y) variables
    X = dataset[:,3:53].astype(float) #X f(eatures) are from the first column and the 50th column
    Y = dataset[:,53] # Y (lables) are from the 50th column
    
    #X = np.random.randn(20, 50)
    #Y = np.random.randn(20)

    print('X shape', X.shape)
    print('Y shape', Y.shape)
    X, Y = trainer.prepare_data(X, Y)
    print('X shape', X.shape)
    print('Y shape', Y.shape)

    model = KerasClassifier(build_fn=trainer.create_model, verbose=1)

    optimizers = ['rmsprop', 'adam']
    init = ['glorot_uniform', 'normal', 'uniform']
    epochs = [20]
    batches = [100]
    hiddens = [2, 5, 7]
    regularizers = [0.001, 0.005, 0.01]
    
    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init, hidden=hiddens, regularize=regularizers)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X, Y)

    print('* Done training')
    
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    

if __name__ == '__main__':
    main()
