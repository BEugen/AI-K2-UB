import iisortrecognize
import cv2
import os
import zlib
import numpy as np

IMG_SOURCES_FOLDERS = '/mnt/data/data/ub-im/AI-K2-UB/img/Class'
IMG_DESTINATION_FOLDERS = '/mnt/data/data/ub-im/AI-K2-UB/img/Sorted'


def main():
    rc = iisortrecognize.RecognizeK2()
    for x in range(0, 5):
        paths = os.listdir(IMG_SOURCES_FOLDERS + '/' + str(x))
        for file in paths:
            im = cv2.resize(cv2.imread(IMG_SOURCES_FOLDERS + '/' + str(x) + '/' + file), (224, 224))
            result, img = rc.recognize(im)
            i = 0
            img_code = zlib.crc32(np.ascontiguousarray(im)) % (1 << 32)
            for c in result:
                if not os.path.exists(IMG_DESTINATION_FOLDERS + '/' + c):
                    os.makedirs(IMG_DESTINATION_FOLDERS + '/' + c)

                fname = str(x) + '_IM-K2-UB_' + str(img_code) + '_' + \
                        str(i) + '_' + c + ".jpg"
                cv2.imwrite(IMG_DESTINATION_FOLDERS + '/' + c + '/' + fname, img[i],
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                i = i + 1
                print(fname)

if __name__ == '__main__':
    main()

