
from PIL import Image
import cv2
import base, recognize
from skimage.measure import compare_ssim
import time
from smb.SMBConnection import SMBConnection


SCORE_STOP = 0.7
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
            bs.savestatistic(classforstatistic(rc_result))
        except Exception as e:
            print(e)

        finally:
            pass

def classforstatistic(data):
    if data['fnn']['-1'] >= SCORE_STOP:
        return STOP_CLASS
    ik = max(data, key=data.get)
    if ik == 0:
        return EMPTY_CLASS
    if ik == 1:
        return IMERROR_CLASS
    return data['snn'] + 2


if __name__ == '__main__':
    main()
