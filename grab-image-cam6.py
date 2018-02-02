import shutil
import tempfile
from datetime import datetime
from PIL import Image
from keras.optimizers import SGD
from keras.models import model_from_json
import cv2, numpy as np

from smb.SMBConnection import SMBConnection



FILE_FOLDER = 'Thumbs'
FILE_NAME = 'cam6.jpg'
STORE_PATH = '/home/administrator/cam6/'

STORE_EMPTY = '/home/administrator/cam6/Empty_NN/'
STORE_FULL = '/home/administrator/cam6/Material_NN/'
STORE_MCLASS = '/home/administrator/cam6/'

MODEL_NAME = '/home/administrator/cam6/script/model_ln'
MODEL_NAME_MCL = '/home/administrator/cam6/script/model_c_ln_1'


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
        conn = SMBConnection(USER_NAME, PASS, CLIENT_NAME, SERVER_NAME, use_ntlm_v2=True)
        conn.connect(SERVER_IP, 139)
        file_obj = tempfile.NamedTemporaryFile()
        conn.retrieveFile(FILE_FOLDER, '/' + FILE_NAME, file_obj)
        file_attributes = conn.getAttributes(FILE_FOLDER, '/' + FILE_NAME)
        file_create_time = datetime.fromtimestamp(file_attributes.last_write_time)
        file_name = CAM_NAME + '_' + \
                    (str(file_create_time.day) if file_create_time.day > 9 else '0' + str(file_create_time.day)) + \
                    (str(file_create_time.month) if file_create_time.month > 9 else '0' + str(file_create_time.month)) + \
                    str(file_create_time.year) + '_' + \
                    (str(file_create_time.hour) if file_create_time.hour > 9 else '0' + str(
                        file_create_time.hour)) + ':' + \
                    (str(file_create_time.minute) if file_create_time.minute > 9 else '0' + str(
                        file_create_time.minute))

        json_file = open(MODEL_NAME + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(MODEL_NAME + '.h5')
        sgd = SGD(lr=0.001, momentum=0.8, decay=0.0, nesterov=False)
        loaded_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

        im = cv2.imread(file_obj.name)
        im = im[21:266, 52:266]
        im = cv2.resize(im, (224, 224))
        im = np.array(im, dtype='float')/255.0
        im = np.expand_dims(im, axis=0)
        (empty, full) = loaded_model.predict(im)[0]
        if empty > full:
            store_image(STORE_EMPTY + file_name, file_obj.name, '0', empty)
        else:
            json_file = open(MODEL_NAME_MCL + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(MODEL_NAME_MCL + '.h5')
            sgd = SGD(lr=0.001, momentum=0.8, decay=0.0, nesterov=False)
            loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            predict = loaded_model.predict(im)[0]
            dct, ind = list_to_dict(predict)
            store_image(STORE_MCLASS + str(ind) + '/' + file_name, file_obj.name, str(ind), dct[ind])

  #      shutil.copy2(file_obj.name, STORE_PATH + file_name)
    except Exception as e:
        print(e)

    finally:
        file_obj.close()


if __name__ == '__main__':
    main()
