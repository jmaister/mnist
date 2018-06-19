import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

from sklearn.preprocessing import MinMaxScaler

import numpy as np
import os


#load data
import cv2


def prepare_image(img):
    resized_image = cv2.resize(img, (28, 28))
    grey_values = 255 - resized_image
    return grey_values

def main():
    X = np.empty((10, 28, 28))
    for i in range(0,10):
        print(i)
        input_image = cv2.imread("./data/test-"+ str(i) +".png", cv2.IMREAD_GRAYSCALE)
        X[i] = prepare_image(input_image)

    #reshape
    print('X', X.shape)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    print('X', X.shape)

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

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        (thresh, bw_img) = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        X = prepare_image(bw_img)
        X = X.reshape(1, X.shape[0], X.shape[1], 1)
        prediction = loaded_model.predict(X)
        num = np.argmax(prediction)
        print('prediction', num)

        cv2.imshow("input", bw_img)

        key = cv2.waitKey(500)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cv2.VideoCapture(0).release()

if __name__ == '__main__':
    main()