import os
import numpy as np
import cv2

IMAGE = {'0': '../img/Class/0/',
         '1': '../img/Class/1/',
         '2': '../img/Class/2/',
         '3': '../img/Class/3/',
         '4': '../img/Class/4/'}
IMAGE_PATH_STORE_TRAIN = '../Imagetrain/train/'
IMAGE_PATH_STORE_TEST = '../Imagetrain/test/'
IMAGE_CLASS = '1'
SPLIT_SIZE = 0.25


def main():
    count = 20000
    for x in IMAGE:
        if int(x) > 2:
            a = 2
        else:
            a = int(x)
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


def convert_image(img):
    kern = build_filters()
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


def build_filters():
    ksize = 11
    theta = np.pi * 0.25
    lamda = np.pi * 0.25
    kern = cv2.getGaborKernel((ksize, ksize), 15.0, theta, lamda, 0.8, 0, ktype=cv2.CV_32F)
    kern /= 1.5 * kern.sum()
    return kern


def img_prepare(path_read, path_store):
    #im= cv2.imread(path_read, cv2.IMREAD_GRAYSCALE)
    #im = cv2.GaussianBlur(im, (3, 3), 0)
    #im = cv2.adaptiveThreshold(im, 255,
    #                          cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 57, 1)
    # (h, w) = im.shape[:2]
    # centr = (w / 3.0, h / 2)
    # M = cv2.getRotationMatrix2D(centr, -15, 1.6)
    # im = cv2.warpAffine(im, M, (w, h))
    # im = im[0:h, 30:190]
    # im = cv2.resize(im, (244, 244))
    # img = cv2.blur(im, (1, 1))
    # img = cv2.adaptiveThreshold(img, 10,
    #                             cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 71, 1)
    # img = cv2.multiply(img, 0.3)
    # img = cv2.blur(img, (1, 1))
    # im = cv2.multiply(im, 0.5)
    # im = cv2.multiply(im, img)
    im = cv2.resize(cv2.imread(path_read, cv2.IMREAD_GRAYSCALE), (224, 224))
    im = convert_image(im)
    cv2.imwrite(path_store, im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    main()
