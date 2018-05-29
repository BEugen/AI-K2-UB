import os
import numpy as np
import cv2

IMAGE = {
         '2': 'Sorted/2/',
         '3': 'Sorted/3/',
         '4': 'Sorted/4/'}
IMAGE_PATH_STORE_TRAIN = 'Imagetrain/train/'
IMAGE_PATH_STORE_TEST = 'Imagetrain/test/'
IMAGE_CLASS = '1'
SPLIT_SIZE = 0.25


def main():
    count = 20000
    for x in IMAGE:
        a = int(x) - 2
        list_file = os.listdir(IMAGE[x])

        split_size = len(list_file) - int(len(list_file) * SPLIT_SIZE)
        print(split_size)
        train = list_file[:split_size]
        test = list_file[split_size:]
        for file in train:
            print(file)
            img_prepare(IMAGE[x] + file, IMAGE_PATH_STORE_TRAIN + 'CAM6_' +
                        str(count) + '_' + str(a) + '.jpg')
            count = count - 1

        for file in test:
            print(file)
            img_prepare(IMAGE[x] + file, IMAGE_PATH_STORE_TEST + 'CAM6_' +
                        str(count) + '_' + str(a) + '.jpg')
            count = count - 1




def img_prepare(path_read, path_store):
    im_b= cv2.imread(path_read, cv2.IMREAD_GRAYSCALE)
    (h, w) = im_b.shape[:2]
    centr = (w / 3, h / 2)
    M = cv2.getRotationMatrix2D(centr, -15, 1.3)
    im_b = cv2.warpAffine(im_b, M, (w, h))
    im_b = im_b[0:h, 30:190]
    im_b = cv2.resize(im_b, (244, 244))
    im = cv2.GaussianBlur(im_b, (1, 1), 0)
    kernel = np.array([[0, -1, 0],
                       [-1, 4.5, -1],
                       [0, -1, 0]])
    im_f = cv2.filter2D(im, -3, kernel, (-5, -5))
    im_ad = cv2.adaptiveThreshold(im_f, 90, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 71, 3)

    struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    im_erod = cv2.erode(im_ad, struct_element)
    im_dilate = cv2.dilate(im_erod, struct_element)
    ot_tr, im_dilate = cv2.threshold(im_dilate, 0, 120, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    hi_tr = ot_tr
    lo_tr = ot_tr * 0.8
    im_can = cv2.Canny(im_dilate, lo_tr, hi_tr)
    im = cv2.addWeighted(im_b, 0.3, im_can, -0.1, 0)
    cv2.imwrite(path_store, im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    main()
