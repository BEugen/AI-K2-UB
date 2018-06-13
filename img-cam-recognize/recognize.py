import numpy as np
import os
import cv2
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.models import Model, load_model
from keras.models import model_from_json
import tensorflow as tf
import itertools
import zlib
import math

sess = tf.Session()
K.set_session(sess)

WEIGHT_RESULT = {'snn1': 1.2, 'snn2': 1.2, 'snn3': 0.8}


class RecognizeK2(object):
    def __init__(self, store=False, store_path=''):
        self.folder_nn = 'nn/'
        self.pfnn = 'model_ln_fnn'
        self.psnn1 = 'model-ub-rnn-ia3'
        self.psnn2 = 'model_ub-ln_snn2'
        self.psnn3 = 'model_ub-ln_snn3'
        self.letters = sorted(list({'0', '1', '2', '3', '4', '5'}))
        self.max_len = 4
        self.fnn = self.get_model_fnn()
        self.snn1 = self.get_model_snn1()
        self.snn2 = self.get_model_snn2()
        self.snn3 = self.get_model_snn3()
        self.store = store
        self.store_path = store_path

    def __snn_result(self, text):
        if '51' in text and '1' in text.replace('51', '', 1):  # bricket
            return 2
        if '55' in text and '1' in text.replace('51', '', 1):  # bricket
            return 2
        if '52' in text and '2' in text.replace('51', '', 1):  # bricket
            return 2
        if '32' in text and '2' in text.replace('32', '', 1):  # dust
            return 0
        if '31' in text and '1' in text.replace('31', '', 1):  # dust
            return 0
        if '22' in text and '2' in text.replace('31', '', 1):  # dust
            return 0
        if '22' in text and '1' in text.replace('31', '', 1):  # dust
            return 0
        return 1  # dust + bricket

    def __snn_calc_result(self, snn):
        rc_snn = []
        for i in range(0, 3):
            t = []
            for x in range(0, 4):
                rc = int(snn[i][x])
                if x > 1:
                    t.append(rc)
            if len(t) > 0:
                rc_snn.append(np.mean(t) * WEIGHT_RESULT['snn' + str(i + 1)])
        rc_calc = math.ceil(np.mean(rc_snn))
        if rc_calc >= 5:
            return 4
        if rc_calc == 4:
            return 3
        if rc_calc >= 2 and rc_calc <= 3 :
            return 2
        #return 2 if result > 2 else result if result > 0 else 0

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

    def __build_filters2(self):
        ksize = 5
        theta = np.pi * 0.25
        lamda = np.pi * 0.25
        kern = cv2.getGaborKernel((ksize, ksize), 15.0, theta, lamda, 0.8, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        return kern

    def __convert_image2(self, img):
        kern = self.__build_filters2()
        mn = np.mean(img)
        if mn > 124:
            k = (mn - 124) * 0.85 / 124 + 0.85
            img = cv2.multiply(img, k)
        if mn < 100:
            k = (mn - 100) * 1.85 / 100 + 1.85
            img = cv2.multiply(img, k)
        img = cv2.medianBlur(img, 7)
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        mn = np.mean(fimg)
        if mn < 80:
            k = (mn - 80) * 1.1 / 80 + 1.1
            fimg = cv2.multiply(fimg, k)
        if mn > 100:
            k = (mn - 100) * 1.1 / 100 + 1.1
            fimg = cv2.multiply(fimg, k)
        return fimg

    def __convert_image3(self, img):
        kern = self.__build_filters2()
        mn = np.mean(img)
        if mn > 124:
            k = (mn - 124) * 0.85 / 124 + 0.85
            img = cv2.multiply(img, k)
        if mn < 100:
            k = (mn - 100) * 1.85 / 100 + 1.85
            img = cv2.multiply(img, k)
        img = cv2.medianBlur(img, 7)
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        mn = np.mean(fimg)
        if mn < 80:
            k = (mn - 80) * 1.1 / 80 + 1.1
            fimg = cv2.multiply(fimg, k)
        if mn > 100:
            k = (mn - 100) * 1.1 / 100 + 1.1
            fimg = cv2.multiply(fimg, k)
        fimg = cv2.adaptiveThreshold(fimg, 10, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 3, 1)
        fimg = cv2.GaussianBlur(fimg, (3, 3), 0)
        return fimg

    def __prepare_img(self, img):
        (h, w) = img.shape[:2]
        point = (w / 3, h / 2)
        M = cv2.getRotationMatrix2D(point, -5, 1.3)
        img = cv2.warpAffine(img, M, (w, h))
        img = img[0:h, 5:w - 5]
        img_nn_1_2 = np.zeros((224, 224 * 4 + 6), np.uint8)
        img_nn_3 = np.zeros((224, 224 * 4), np.uint8)
        for x in range(0, 4):
            y1, y2, x1, x2 = self.__section_img(x)
            im = cv2.resize(img[y1:y2, x1:x2], (224, 224))
            img_nn_3[0:224, x * 224:x * 224 + 224] = self.__convert_image3(im)
            if x == 0:
                img_nn_1_2[0:224, 0:224] = self.__convert_image2(im)
            else:
                img_nn_1_2[0:224, x * 226:x * 226 + 224] = self.__convert_image2(im)
        return img_nn_1_2, img_nn_3

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
        nn_result = {}
        image = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2GRAY)
        im_p = self.__convert_image(image)
        im_p = np.array(im_p, dtype='float') / 255.0
        im_p = np.expand_dims(im_p, axis=-1)
        im_p = np.expand_dims(im_p, axis=0)
        dict, ind = self.list_to_dict(self.fnn.predict(im_p)[0])
        nn_result['fnn'] = dict
        if ind == 2:
            img_1_2, img_3 = self.__prepare_img(image)
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
                    snn1text = snn1text + '1'
            if ln > 4:
                snn1text = snn1text[0:4]
            pred_texts = []
            pred_texts.append(snn1text)
            pred_texts.append('')
            pred_texts.append('')
            for x in range(0, 4):
                im_nn_2 = img_1_2[0:224, x * 226:x * 226 + 224]
                im_nn_3 = img_3[0:224, x * 224:x * 224 + 224]
                im_nn_2 = np.array(im_nn_2, dtype='float') / 255.0
                im_nn_3 = np.array(im_nn_3, dtype='float') / 255.0
                im_nn_2 = np.expand_dims(im_nn_2, axis=0)
                im_nn_2 = np.expand_dims(im_nn_2, axis=-1)
                im_nn_3 = np.expand_dims(im_nn_3, axis=0)
                im_nn_3 = np.expand_dims(im_nn_3, axis=-1)
                dict, ind = self.list_to_dict(self.snn2.predict(im_nn_2)[0])
                pred_texts[1] = pred_texts[1] + str(ind)
                dict, ind = self.list_to_dict(self.snn3.predict(im_nn_3)[0])
                pred_texts[2] = pred_texts[2] + str(ind)
            # narr = np.zeros(3)
            for i in range(0, 3):
                key = 'snn' + str(i + 1)
                nn_result[key] = pred_texts[i]
            #    narr[i] = self.__snn_result(pred_texts[i]) * WEIGHT_RESULT[key]
            nn_result['snn'] = self.__snn_calc_result(pred_texts)  # round(narr.mean(), 0)
        if self.store:
            if ind == 2:
                pt = self.store_path + '/' + str(int(nn_result['snn'] + 2))
                nc = str(int(nn_result['snn'] + 2))
            else:
                pt = self.store_path + '/' + str(ind)
                nc = str(ind)
            if not os.path.exists(pt):
                os.makedirs(pt)
            img_code = zlib.crc32(np.ascontiguousarray(image)) % (1 << 32)
            cv2.imwrite(pt + '/CAM_' + str(img_code) + '_' + nc + '.jpg',
                        image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return nn_result

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

    def get_model_snn2(self):
        json_file = open(self.folder_nn + self.psnn2 + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.folder_nn + self.psnn2 + '.h5')
        sgd = SGD(lr=0.001, momentum=0.8, decay=0.0, nesterov=False)
        loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return loaded_model

    def get_model_snn3(self):
        json_file = open(self.folder_nn + self.psnn3 + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.folder_nn + self.psnn3 + '.h5')
        sgd = SGD(lr=0.001, momentum=0.8, decay=0.0, nesterov=False)
        loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return loaded_model

    def get_model_fnn(self):
        json_file = open(self.folder_nn + self.pfnn + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.folder_nn + self.pfnn + '.h5')
        sgd = SGD(lr=0.001, momentum=0.8, decay=0.0, nesterov=False)
        loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
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
