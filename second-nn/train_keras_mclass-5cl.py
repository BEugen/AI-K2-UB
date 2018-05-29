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
import random
from keras import callbacks
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


IMG_PATH_TRAIN = '/mnt/data/data/ub-im/Imgtrain/strain/'
IMG_PATH_TEST= '/mnt/data/data/ub-im/Imgtrain/stest/'
#IMG_PATH = 'Imgtrain/'
BATCH_SIZE = 64
NB_EPOCH = 100
NB_CLASSES = 2
VERBOSE = 1
VALIDATION_SPLIT = 0.25
OPTIM = SGD()#Adam(lr=INIT_LR, decay=INIT_LR / NB_EPOCH)


FOLDER_RESULT = ''

def LeNet():
    model = Sequential()
    # CONV => RELU => POOL
    model.add(Conv2D(20, kernel_size=3, padding="same",
                     input_shape=(224, 224, 1), kernel_initializer='he_normal'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.20))
    # CONV => RELU => POOL
    model.add(Conv2D(50, kernel_size=5, padding="same", kernel_initializer='he_normal'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(70, kernel_size=3, padding="same", kernel_initializer='he_normal'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(50, kernel_size=5, padding="same", kernel_initializer='he_normal'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(70, kernel_size=3, padding="same", kernel_initializer='he_normal'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    model.add(Dropout(0.30))
    # Flatten => RELU layers
    model.add(Flatten())
    model.add(Dense(500, kernel_initializer='he_normal'))
    model.add(Activation("relu"))

    # a softmax classifier
    model.add(Dense(6))
    model.add(Activation("softmax"))

    return model

def LeNetKg():
    model = Sequential()
    model.add(Conv2D(12, 5, 5, activation='relu', input_shape=(224, 224, 1), init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(25, 5, 5, activation='relu', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(180, activation='relu', init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu', init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax', init='he_normal'))
    return model

def load_image(path):
    list_file = os.listdir(path)
    random.seed(30)
    random.shuffle(list_file)
    x_data = []
    y_data = []
    for file in list_file:
        flabel = os.path.splitext(file)[0].split('_')
        im = cv2.resize(cv2.imread(path + file,  cv2.IMREAD_GRAYSCALE), (224, 224))
        im = img_to_array(im)
        x_data.append(im)
        y_data.append(flabel[len(flabel)-1])
    x_data = np.array(x_data, dtype='float')/255.0
    y_data = np.array(y_data)
    return x_data, y_data


def main(args):
    X_train, Y_train = load_image(IMG_PATH_TRAIN)
    Y_train = np_utils.to_categorical(Y_train, num_classes=6)
    X_test, Y_test = load_image(IMG_PATH_TEST)
    Y_test= np_utils.to_categorical(Y_test, num_classes=6)
    #(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=.25, random_state=40)

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}_snn_ic.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    #lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    #tensorboard = TensorBoard(log_dir="/home/eugen/PycharmProjects/AI-K2-UB/logs/{}".format(time()), write_graph=True,
    #                          write_grads=True, write_images=True,
    #                          histogram_freq=1)
    # fit
    model = LeNet()
    model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=VERBOSE,
                        validation_data=(X_test, Y_test),
                        validation_split=VALIDATION_SPLIT, callbacks=[log, tb, checkpoint])

    score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
    print('Test score:', score[0])
    print('Test accuracy', score[1])

    # save model
    model_json = model.to_json()
    with open(args.save_dir + "/model_ln_snn_ic.json", "w") as json_file:
        json_file.write(model_json)
    #model.save_weights("model_ln_snn_ic.h5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    main(args)
