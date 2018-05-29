import os
import numpy as np
import cv2

IMAGE = {
         '0': '/mnt/data/data/ub-im/Sort-result/0/',
         '1': '/mnt/data/data/ub-im/Sort-result/1/',
         '2': '/mnt/data/data/ub-im/Sort-result/2/',
         '3': '/mnt/data/data/ub-im/Sort-result/3/',
         '4': '/mnt/data/data/ub-im/Sort-result/4/'}
IMAGE_PATH_STORE_TRAIN = '/mnt/data/data/ub-im/Imgtrain/strain/'
IMAGE_PATH_STORE_TEST = '/mnt/data/data/ub-im/Imgtrain/stest/'
IMAGE_CLASS = '1'
SPLIT_SIZE = 0.25


def main():
    for x in IMAGE:
        list_file = os.listdir(IMAGE[x])

        split_size = len(list_file) - int(len(list_file) * SPLIT_SIZE)
        print(split_size)
        train = list_file[:split_size]
        test = list_file[split_size:]
        for file in train:
            print(file)
            img_prepare(IMAGE[x] + file, IMAGE_PATH_STORE_TRAIN + file)

        for file in test:
            print(file)

            img_prepare(IMAGE[x] + file, IMAGE_PATH_STORE_TEST + file)

def build_filters():
    ksize = 11
    theta = np.pi*0.25
    lamda = np.pi*0.25
    kern = cv2.getGaborKernel((ksize, ksize), 15.0, theta, lamda, 0.8, 0, ktype=cv2.CV_32F)
    kern /= 1.5*kern.sum()
    return kern


def img_prepare(path_read, path_store):
    im= cv2.imread(path_read, cv2.IMREAD_GRAYSCALE)
    # im = cv2.GaussianBlur(im_b, (1, 1), 0)
    # kernel = np.array([[0, -1, 0],
    #                    [-1, 4.5, -1],
    #                    [0, -1, 0]])
    # im_f = cv2.filter2D(im, -3, kernel, (-5, -5))
    # im_ad = cv2.adaptiveThreshold(im_f, 90, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 71, 3)
    #
    # struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # im_erod = cv2.erode(im_ad, struct_element)
    # im_dilate = cv2.dilate(im_erod, struct_element)
    # ot_tr, im_dilate = cv2.threshold(im_dilate, 0, 120, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # hi_tr = ot_tr
    # lo_tr = ot_tr * 0.8
    # im_can = 255 - cv2.Canny(im_dilate, lo_tr, hi_tr)
    # im_can = cv2.GaussianBlur(im_can, (3, 3), 0)
    kern = build_filters()
    mn = np.mean(im)
    if mn > 130:
        k = (mn - 130) * 0.85 / 130 + 0.85
        im = cv2.multiply(im, k)
    if mn < 110:
        k = (mn - 110) * 1.85 / 110 + 1.85
        im = cv2.multiply(im, k)
    im = cv2.medianBlur(im, 9)
    fimg = cv2.filter2D(im, cv2.CV_8UC3, kern)
    mn = np.mean(fimg)
    if mn < 80:
        k = (mn - 80) * 1.2 / 80 + 1.2
        fimg = cv2.multiply(fimg, k)
    ret, im_ad = cv2.threshold(fimg, 60, 200, cv2.THRESH_BINARY)
    ret, im_ad1 = cv2.threshold(fimg, 200, 60, cv2.THRESH_BINARY_INV)
    im_ad = 255 - cv2.multiply(im_ad, im_ad1)
    im_ad = cv2.GaussianBlur(im_ad, (13, 13), 0)
#    struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#    im_erod = cv2.erode(im_ad, struct_element)
#    im_dilate = cv2.dilate(im_erod, struct_element)
#    ot_tr, im_dilate = cv2.threshold(im_dilate, 40, 150, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#    hi_tr = ot_tr
#    lo_tr = ot_tr * 0.3
#    im_can = cv2.Canny(im_dilate, lo_tr, hi_tr)
    cv2.imwrite(path_store, im_ad, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    main()
