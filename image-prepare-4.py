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
    im = cv2.imread(path_read, cv2.IMREAD_GRAYSCALE)
    #img = cv2.blur(im, (1, 1))
    #img = cv2.adaptiveThreshold(img, 10,
    #                            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 1)
    #im = np.multiply(im, img*0.3)
    #im = cv2.blur(im, (3, 3))
    #-----------------------
    #img = cv2.blur(im, (1, 1))
    #img = cv2.adaptiveThreshold(img, 10,
    #                            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 1)
    #img = cv2.multiply(img, 0.2)
    #img = cv2.blur(img, (5, 5))
    #im = cv2.multiply(im, 0.5)
    #im = cv2.multiply(im, img)
    im_g = cv2.GaussianBlur(im, (3, 3), 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    im_f = cv2.filter2D(im_g, -1, kernel, (-1, -1))
    im_wh = cv2.addWeighted(im_f, 1.5, im, -0.5, 0)
    im_ad = cv2.adaptiveThreshold(im_wh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 20)
    struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    im_erod = cv2.erode(im_ad, struct_element)
    im_dilate = cv2.dilate(im_erod, struct_element)
    ot_tr, im_dilate = cv2.threshold(im_dilate, 0, 127, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    hi_tr = ot_tr
    lo_tr = ot_tr * 0.5
    im_can = cv2.Canny(im_dilate, lo_tr, hi_tr)
    cv2.imwrite(path_store, im_can, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    main()

