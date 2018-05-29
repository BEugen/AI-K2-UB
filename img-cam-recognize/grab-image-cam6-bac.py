from datetime import datetime
from PIL import Image
from keras.optimizers import SGD
from keras.models import model_from_json
import cv2, numpy as np
import base
import os
import random
from skimage.measure import compare_ssim
import time

#from smb.SMBConnection import SMBConnection

CAM_NAME = ''
USER_NAME = ''
PASS = ''
CLIENT_NAME = ''
SERVER_NAME = ''
SERVER_IP = ''

FILE_FOLDER = 'Imagetrain/train'

MODEL_NAME = '3class-2nn/model_ln_fnn'
MODEL_NAME_MCL = '3class-2nn/model_ln'
CLASS_STOP = -1


def store_image(path_save, path_load, class_number, pred):
    img = Image.open(path_load)
    img = img.crop((21, 52, 266, 266))
    img.save(path_save + '_' + str(round(pred * 100, 2)) + '_' + class_number + '.jpg')


def list_to_dict(li):
    dct = {}
    ind = 0
    mx = 0.0
    ind_max = 0
    for item in li:
        dct[ind] = item
        if item > mx:
            ind_max = ind
            mx = item
        ind = ind + 1
    return dct, ind_max


def main():
    try:
        while True:
            # conn = SMBConnection(USER_NAME, PASS, CLIENT_NAME, SERVER_NAME, use_ntlm_v2=True)
            # conn.connect(SERVER_IP, 139)
            # file_obj = tempfile.NamedTemporaryFile()
            # conn.retrieveFile(FILE_FOLDER, '/' + FILE_NAME, file_obj)
            # file_attributes = conn.getAttributes(FILE_FOLDER, '/' + FILE_NAME)
            # file_create_time = datetime.fromtimestamp(file_attributes.last_write_time)
            # file_name = CAM_NAME + '_' + \
            #             (str(file_create_time.day) if file_create_time.day > 9 else '0' + str(file_create_time.day)) + \
            #             (str(file_create_time.month) if file_create_time.month > 9 else '0' + str(file_create_time.month)) + \
            #             str(file_create_time.year) + '_' + \
            #             (str(file_create_time.hour) if file_create_time.hour > 9 else '0' + str(
            #                 file_create_time.hour)) + ':' + \
            #             (str(file_create_time.minute) if file_create_time.minute > 9 else '0' + str(
            #                 file_create_time.minute))

            bs = base.Psql()
            path = FILE_FOLDER + '/' + random.choice(os.listdir(FILE_FOLDER))
            im = cv2.imread(path)
            file_attributes = os.stat(path)
            file_create_time = datetime.fromtimestamp(file_attributes.st_ctime)
            #shutil.copy2(file_obj.name, STORE_PATH + 'img/' + file_name + '.jpg')
            #im = im[72:316, 37:241]

            json_file = open(MODEL_NAME + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(MODEL_NAME + '.h5')
            sgd = SGD(lr=0.001, momentum=0.8, decay=0.0, nesterov=False)
            loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            im_l, guid, fd = bs.loadlast()
            #if file_create_time == fd:
            #    return
            score = 1.0
            test = len(im_l)
            if im_l is not None and len(im_l) > 0:
                grayA = cv2.cvtColor(cv2.resize(im, (224, 224)), cv2.COLOR_BGR2GRAY)
                grayB = cv2.cvtColor(cv2.resize(im_l, (224, 224)), cv2.COLOR_BGR2GRAY)
                (score, diff) = compare_ssim(grayA, grayB, full=True)
                if score > 0.7:
                    bs.updatelast(guid, score)
                    bs.savestatistic(CLASS_STOP)
                    break
            im_p = cv2.cvtColor(cv2.resize(im, (224, 224)), cv2.COLOR_BGR2GRAY)
            im_p = np.array(im_p, dtype='float') / 255.0
            im_p = np.expand_dims(im_p, axis=2)
            im_p = np.expand_dims(im_p, axis=0)
            (empty, noimage, full) = loaded_model.predict(im_p)[0]
            guid = bs.save(im, score, empty, full, file_create_time)
            dct, ind = list_to_dict((empty, noimage, full))

            if ind == 0 or ind == 1: #empty or no image recognise
                #store_image(STORE_EMPTY + file_name, file_obj.name, '0', empty)
                bs.savecategory1(guid, ind, dct)
                bs.savestatistic(ind)
            else:
                json_file = open(MODEL_NAME_MCL + '.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = model_from_json(loaded_model_json)
                loaded_model.load_weights(MODEL_NAME_MCL + '.h5')
                sgd = SGD(lr=0.001, momentum=0.8, decay=0.0, nesterov=False)
                loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

                (h, w) = im.shape[:2]
                centr = (w / 3.0, h / 2)
                M = cv2.getRotationMatrix2D(centr, -15, 1.6)
                im = cv2.warpAffine(im, M, (w, h))
                im = im[0:h, 30:190]
                im = cv2.cvtColor(cv2.resize(im, (224, 224)), cv2.COLOR_BGR2GRAY)
                img = cv2.blur(im, (1, 1))
                img = cv2.adaptiveThreshold(img, 10,
                                            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 71, 1)
                img = cv2.multiply(img, 0.3)
                img = cv2.blur(img, (1, 1))
                im = cv2.multiply(im, 0.5)
                im = cv2.multiply(im, img)
                im_p = np.array(im, dtype='float') / 255.0
                im_p = np.expand_dims(im_p, axis=2)
                im_p = np.expand_dims(im_p, axis=0)
                predict = loaded_model.predict(im_p)[0]
                dct, ind = list_to_dict(predict)
                ind = ind + 2
                bs.savecategory2(guid, ind, dct)
                bs.savestatistic(ind)
                #store_image(STORE_MCLASS + str(ind) + '/' + file_name, file_obj.name, str(ind), dct[ind])

        #      shutil.copy2(file_obj.name, STORE_PATH + file_name)
                time.sleep(30)
    except Exception as e:
        print(e)

    finally:
        pass


if __name__ == '__main__':
    main()
