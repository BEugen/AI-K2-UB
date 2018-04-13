from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
import cv2, numpy as np
import os
import keras
from keras.callbacks import TensorBoard
from time import time
import random

IMG_PATH_TRAIN = 'Imagetrain/train/'
IMG_PATH_TEST= 'Imagetrain/test/'
#IMG_PATH = 'Imgtrain/'
BATCH_SIZE = 128
NB_EPOCH = 100
NB_CLASSES = 2
VERBOSE = 1
VALIDATION_SPLIT = 0.25
INIT_LR = 1e-3
OPTIM = SGD()#Adam(lr=INIT_LR, decay=INIT_LR / NB_EPOCH)





def load_image(path):
    list_file = os.listdir(path)
    random.seed(40)
    random.shuffle(list_file)
    x_data = []
    y_data = []
    for file in list_file:
        flabel = os.path.splitext(file)[0].split('_')
        im = cv2.resize(cv2.imread(path + file), (224, 224))
        im = img_to_array(im)
        x_data.append(im)
        y_data.append(flabel[len(flabel)-1])
    x_data = np.array(x_data, dtype='float')/255.0
    y_data = np.array(y_data)
    return x_data, y_data


def main():
    X_train, Y_train = load_image(IMG_PATH_TRAIN)
    Y_train = np_utils.to_categorical(Y_train, num_classes=3)
    X_test, Y_test = load_image(IMG_PATH_TEST)
    Y_test= np_utils.to_categorical(Y_test, num_classes=3)
    #(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=.25, random_state=40)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()), write_graph=True, write_grads=True, write_images=True,
                              histogram_freq=0)
    # fit
    model = keras.applications.resnet50.ResNet50(include_top=True, input_tensor=None, input_shape=None, pooling=None, classes=3)
    model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,
                        validation_data=(X_test, Y_test),
                        validation_split=VALIDATION_SPLIT, callbacks=[tensorboard])

    score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
    print('Test score:', score[0])
    print('Test accuracy', score[1])

    # save model
    model_json = model.to_json()
    with open("model_ln_2.json", "w") as json_file:
        json_file.write(model_json)
        #serialize weights to HDF5
    model.save_weights("model_ln_2.h5")


if __name__ == '__main__':
    main()
