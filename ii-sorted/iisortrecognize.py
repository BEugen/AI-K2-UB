import numpy as np
import time
import numpy as np
import os
import json
import shutil
import cv2
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model
from keras.layers.recurrent import GRU
from time import time
from keras.models import model_from_json
import tensorflow as tf
import itertools
import zlib

sess = tf.Session()
K.set_session(sess)

WEIGHT_RESULT = {'snn1': 1.5, 'snn2': 1.0, 'snn3': 0.7}


class RecognizeK2(object):
    def __init__(self, store=False, store_path=''):
        self.folder_nn = 'nn/'
        self.psnn1 = 'model-ub-rnn'
        self.letters = sorted(list({'0', '1', '2', '3', '4', '5'}))
        self.max_len = 4
        self.snn1 = self.get_model_snn1()
        self.store = store
        self.store_path = store_path
        self.img = []

    def __convert_image(self, img):
        kern = self.__build_filters()
        mn = np.mean(img)
        if mn > 130:
            k = (mn - 130) * 0.85 / 130 + 0.85
            img = cv2.multiply(img, k)
        if mn < 110:
            k = (mn - 110) * 1.85 / 110 + 1.85
            img = cv2.multiply(img, k)
        img = cv2.medianBlur(img, 9)
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        mn = np.mean(fimg)
        if mn < 80:
            k = (mn - 80) * 1.2 / 80 + 1.2
            fimg = cv2.multiply(fimg, k)
        ret, im_ad = cv2.threshold(fimg, 60, 200, cv2.THRESH_BINARY)
        ret, im_ad1 = cv2.threshold(fimg, 200, 60, cv2.THRESH_BINARY_INV)
        im_ad = 255 - cv2.multiply(im_ad, im_ad1)
        im_ad = cv2.GaussianBlur(im_ad, (13, 13), 0)
        return im_ad

    def __build_filters(self):
        ksize = 11
        theta = np.pi * 0.25
        lamda = np.pi * 0.25
        kern = cv2.getGaborKernel((ksize, ksize), 15.0, theta, lamda, 0.8, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        return kern

    def __prepare_img(self, img):
        (h, w) = img.shape[:2]
        point = (w / 3, h / 2)
        M = cv2.getRotationMatrix2D(point, -5, 1.3)
        img = cv2.warpAffine(img, M, (w, h))
        img = img[0:h, 5:w-5]
        img_nn_1_2 = np.zeros((224, 224 * 4 + 6), np.uint8)
        self.img = []
        for x in range(0, 4):
            y1, y2, x1, x2 = self.__section_img(x)
            im = cv2.resize(img[y1:y2, x1:x2], (224, 224))
            self.img.append(im)
            if x == 0:
                img_nn_1_2[0:224, 0:224] = im
            else:
                img_nn_1_2[0:224, x * 226:x * 226 + 224] = im
        return img_nn_1_2

    def __section_img(self, index):
        if index == 0:
            # |x| |
            # | | |
            return 0, 112, 0, 112
        elif index == 1:
            # | |x|
            # | | |
            return 0, 112, 112, 224
        elif index == 2:
            # | | |
            # |x| |
            return 112, 224, 0, 112
        else:
            # | | |
            # | |x|
            return 112, 224, 112, 224

    def recognize(self, image):
        image = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2GRAY)
        img_1_2 = self.__prepare_img(image)
        net_inp = self.snn1.get_layer(name='the_input').input
        net_out = self.snn1.get_layer(name='softmax').output
        im_d = np.array(img_1_2, dtype='float') / 255.0
        im_d = im_d.T
        im_d = np.expand_dims(im_d, -1)
        im_d = np.expand_dims(im_d, 0)
        net_out_value = sess.run(net_out, feed_dict={net_inp: im_d})
        snn1text = self.decode_batch(net_out_value)[0]
        ln = len(snn1text)
        if ln < 4:
            for i in range(0, 4 - ln):
                snn1text = snn1text + 'n'
        if ln > 4:
            snn1text = snn1text[0:4]
        return snn1text, self.img


    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def decode_batch(self, out):
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = ''
            for c in out_best:
                if c < len(self.letters):
                    outstr += self.letters[c]
            ret.append(outstr)
        return ret

    def get_model_snn1(self):
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        loaded_model = load_model(self.folder_nn + self.psnn1 + '.h5', compile=False)
        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        loaded_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
        return loaded_model



    def list_to_dict(self, li):
        dct = {}
        ind = 0
        mx = 0.0
        ind_max = 0
        for item in li:
            dct[str(ind)] = item
            if item > mx:
                ind_max = ind
                mx = item
            ind = ind + 1
        return dct, ind_max
