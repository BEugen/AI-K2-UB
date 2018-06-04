import os
from PIL import Image
from keras.optimizers import SGD
from keras.models import model_from_json
import cv2, numpy as np
import base, recognize
from skimage.measure import compare_ssim
import random
import time

# from smb.SMBConnection import SMBConnection


FILE_FOLDER = '/mnt/data/data/ub-im/Sorted/2'

MODEL_NAME = '3class-2nn/model_ln_fnn'
MODEL_NAME_MCL = '3class-2nn/model_ln'

SCORE_STOP = 0.7
STOP_CLASS = -1
EMPTY_CLASS = 0
DUST_CLASS = 2
IMERROR_CLASS = 1
BRBRIKET_CLASS = 3
BRIKET_CLASS = 4

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
    rc = recognize.RecognizeK2()
    bs = base.Psql()
    while True:
        time.sleep(25)
        try:
            conn = SMBConnection(USER_NAME, PASS, CLIENT_NAME, SERVER_NAME, use_ntlm_v2=True)
            conn.connect(SERVER_IP, 139)
            file_obj = tempfile.NamedTemporaryFile()
            conn.retrieveFile(FILE_FOLDER, '/' + FILE_NAME, file_obj)
            file_attributes = conn.getAttributes(FILE_FOLDER, '/' + FILE_NAME)
            file_create_time = datetime.fromtimestamp(file_attributes.last_write_time)
            file_name = CAM_NAME + '_' + \
                        (str(file_create_time.day) if file_create_time.day > 9 else '0' + str(file_create_time.day)) + \
                        (str(file_create_time.month) if file_create_time.month > 9 else '0' + str(
                            file_create_time.month)) + \
                        str(file_create_time.year) + '_' + \
                        (str(file_create_time.hour) if file_create_time.hour > 9 else '0' + str(
                            file_create_time.hour)) + ':' + \
                        (str(file_create_time.minute) if file_create_time.minute > 9 else '0' + str(
                            file_create_time.minute))
            im = cv2.imread(file_obj.name)
            x1, x2, y1, y2 = bs.getcropimg()
            if x1 is None:
                continue
            im = im[y1:y2, x1:x2]
            rc_result = rc.recognize(im)
            im_l, img_guid, img_tfile = bs.loadimglast()
            if file_create_time == img_tfile:
                continue
            if im_l is not None and len(im_l) > 0:
                grayA = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                grayB = cv2.cvtColor(im_l, cv2.COLOR_BGR2GRAY)
                (score, diff) = compare_ssim(grayA, grayB, full=True)
                rc_result['fnn']['-1'] = score
                if score > SCORE_STOP:
                    bs.updateimglast(img_guid, score)
                    continue

            for x in range(1, 4):
                if 'snn' + str(x) is not rc_result.keys():
                    rc_result['snn' + str(x)] = ''
            bs.savedata(im, rc_result, file_create_time)
            bs.savestatistic(ind)
        except Exception as e:
            print(e)

        finally:
            pass

def clssforstatisitc(data):
    if data['fnn']['-1'] >= SCORE_STOP:
        return STOP_CLASS


if __name__ == '__main__':
    main()
