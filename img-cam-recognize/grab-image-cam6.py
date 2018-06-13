
import cv2
import base, recognize, video_capture
from skimage.measure import compare_ssim
import time
from smb.SMBConnection import SMBConnection
import tempfile
from datetime import datetime

CAM_NAME = 'CAM6'
USER_NAME = 'Admin'
PASS = '111'
CLIENT_NAME = 'skzasutp0004'
SERVER_NAME = 'VIDEOSRV-1'
SERVER_IP = '192.168.0.1'

FILE_FOLDER = 'Thumbs'
FILE_NAME = 'cam6.jpg'
STORE_PATH = '/home/administrator/cam6/'

SCORE_STOP = 0.85
STOP_CLASS = -1
EMPTY_CLASS = 0
DUST_CLASS = 2
IMERROR_CLASS = 1
BRBRIKET_CLASS = 3
BRIKET_CLASS = 4


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
    rc = recognize.RecognizeK2(store=True, store_path='img')
    bs = base.Psql()
    vc = video_capture.VideoCap()
    vc.start_capture('rtsp://admin:123456Qw@192.168.0.28:554/Streaming/channels/102')
    while True:
        time.sleep(10)
        try:
            im, file_create_time = vc.image()
            x1, x2, y1, y2 = bs.getcropimg()
            if x1 is None:
                continue
            (h, w) = im.shape[:2]
            point = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(point, -5, 1.0)
            im = cv2.warpAffine(im, M, (w, h))
            im = im[y1:y2, x1:x2]
            print('recognize')
            rc_result = rc.recognize(im)
            print(rc_result)
            im_l, img_guid, img_tfile = bs.loadimglast()
            if file_create_time == img_tfile:
                continue
            if im_l is not None and len(im_l) > 0:
                grayA = cv2.cvtColor(cv2.resize(im, (224, 224)), cv2.COLOR_BGR2GRAY)
                grayB = cv2.cvtColor(cv2.resize(im_l, (224, 224)), cv2.COLOR_BGR2GRAY)
                (score, diff) = compare_ssim(grayA, grayB, full=True)
                rc_result['fnn']['-1'] = score
                print(score)
                if score > SCORE_STOP:
                    bs.updateimglast(img_guid, score)
                    bs.savestatistic(classforstatistic(rc_result))
                    continue
            for x in range(1, 4):
                if 'snn' + str(x) in rc_result.keys():
                    continue
                else:
                    rc_result['snn' + str(x)] = ''
            rc_result['snn'] = classforstatistic(rc_result)
            bs.savedata(im, rc_result, file_create_time)
            bs.savestatistic(rc_result['snn'])
            print(rc_result)
        except Exception as e:
            print(e)

        finally:
            pass
            #conn.close()

def classforstatistic(data):
    if data['fnn']['-1'] >= SCORE_STOP:
        return STOP_CLASS
    ik = int(max(data['fnn'], key=data['fnn'].get))
    if ik == 0:
        return EMPTY_CLASS
    if ik == 1:
        return IMERROR_CLASS
    return data['snn']


if __name__ == '__main__':
    main()
