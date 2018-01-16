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
from keras.callbacks import TensorBoard
from time import time

#IMG_PATH = '/home/administrator/cam6/Imgtrain/'
IMG_PATH = '/home/administrator/projects/aikuub/Imgtrain/'
BATCH_SIZE = 32
NB_EPOCH = 20
NB_CLASSES = 2
VERBOSE = 1
VALIDATION_SPLIT = 0.2
INIT_LR = 1e-3
OPTIM = Adam(lr=INIT_LR, decay=INIT_LR / NB_EPOCH)


def VGG_16():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model


def load_image(path):
    list_file = os.listdir(path)
    x_data = []
    y_data = []
    for file in list_file:
        flabel = os.path.splitext(file)[0].split('_')
        im = cv2.resize(cv2.imread(path + file), (224, 224))
        im = img_to_array(im)
        x_data.append(im)
        y_data.append(flabel[3])
    x_data = np.array(x_data, dtype='float')/255.0
    y_data = np.array(y_data)
    return x_data, y_data


def main():
    X, Y = load_image(IMG_PATH)
    Y = np_utils.to_categorical(Y, num_classes=2)
    (X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=.25, random_state=40)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()), write_graph=True, write_grads=True, write_images=True,
                              histogram_freq=0)
    # fit
    model = VGG_16()
    model.compile(loss='binary_crossentropy', optimizer=OPTIM, metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,
                        validation_data=(X_test, Y_test),
                        validation_split=VALIDATION_SPLIT, callbacks=[tensorboard])

    score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
    print('Test score:', score[0])
    print('Test accuracy', score[1])
    print(history.history.keys)

    # save model
    model_json = model.to_json()
    with open("model_n.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_n.h5")


if __name__ == '__main__':
    main()
