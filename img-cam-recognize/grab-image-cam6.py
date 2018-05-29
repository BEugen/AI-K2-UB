import os
from PIL import Image
from keras.optimizers import SGD
from keras.models import model_from_json
import cv2, numpy as np
import base, recognize
from skimage.measure import compare_ssim
import random
#from smb.SMBConnection import SMBConnection



FILE_FOLDER = '/mnt/data/data/ub-im/Sorted/2'

MODEL_NAME = '3class-2nn/model_ln_fnn'
MODEL_NAME_MCL = '3class-2nn/model_ln'



def store_image(path_save, path_load, class_number, pred):
    img = Image.open(path_load)
    img = img.crop((21, 52, 266, 266))
    img.save(path_save + '_' + str(round(pred*100, 2)) + '_' + class_number + '.jpg')


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
        #conn = SMBConnection(USER_NAME, PASS, CLIENT_NAME, SERVER_NAME, use_ntlm_v2=True)
        #conn.connect(SERVER_IP, 139)
        #file_obj = tempfile.NamedTemporaryFile()
        #conn.retrieveFile(FILE_FOLDER, '/' + FILE_NAME, file_obj)
        #file_attributes = conn.getAttributes(FILE_FOLDER, '/' + FILE_NAME)
        #file_create_time = datetime.fromtimestamp(file_attributes.last_write_time)
        #file_name = CAM_NAME + '_' + \
        #            (str(file_create_time.day) if file_create_time.day > 9 else '0' + str(file_create_time.day)) + \
        #            (str(file_create_time.month) if file_create_time.month > 9 else '0' + str(file_create_time.month)) + \
        #            str(file_create_time.year) + '_' + \
        #            (str(file_create_time.hour) if file_create_time.hour > 9 else '0' + str(
        #                file_create_time.hour)) + ':' + \
        #            (str(file_create_time.minute) if file_create_time.minute > 9 else '0' + str(
        #                file_create_time.minute))
        #
        rc = recognize.RecognizeK2()
        bs = base.Psql()
        imfile = random.choice(os.listdir(FILE_FOLDER))
        im = cv2.imread(FILE_FOLDER + '/' + imfile)
        print(im.shape)
        rc.recognize(im)

        json_file = open(MODEL_NAME + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(MODEL_NAME + '.h5')
        sgd = SGD(lr=0.001, momentum=0.8, decay=0.0, nesterov=False)
        loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        im_l, guid = bs.loadlast()
        grayA = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(im_l, cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        if score > 0.8:
            bs.updatelast(guid)
            return

        im_p = cv2.resize(im, (224, 224))
        im_p = np.array(im_p, dtype='float')/255.0
        im_p = np.expand_dims(im_p, axis=0)
        (empty, full) = loaded_model.predict(im_p)
        guid = bs.save(im, score, empty, full)
        json_file = open(MODEL_NAME_MCL + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(MODEL_NAME_MCL + '.h5')
        sgd = SGD(lr=0.001, momentum=0.8, decay=0.0, nesterov=False)
        loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        predict = loaded_model.predict(im)[0]
        dct, ind = list_to_dict(predict)
        bs.savecategory(guid, ind, dct)
        bs.savestatistic(ind)

    except Exception as e:
        print(e)

    finally:
        pass


if __name__ == '__main__':
    main()
