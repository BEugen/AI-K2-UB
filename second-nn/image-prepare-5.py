import os
import numpy as np
import cv2

IMAGE = {
         '0': '/mnt/data/data/ub-im/AI-K2-UB/img/Sorted/0/',
         '1': '/mnt/data/data/ub-im/AI-K2-UB/img/Sorted/1/',
         '2': '/mnt/data/data/ub-im/AI-K2-UB/img/Sorted/2/',
         '3': '/mnt/data/data/ub-im/AI-K2-UB/img/Sorted/3/',
         '4': '/mnt/data/data/ub-im/AI-K2-UB/img/Sorted/4/',
         '5': '/mnt/data/data/ub-im/AI-K2-UB/img/Sorted/5/'}
IMAGE_PATH_STORE_TRAIN = '/mnt/data/data/ub-im/Imgtrain/ltrain/'
IMAGE_PATH_STORE_TEST = '/mnt/data/data/ub-im/Imgtrain/ltest/'
IMAGE_CLASS = '1'
SPLIT_SIZE = 0.25


def main():
    for x in range(0, 6):
        list_file = os.listdir(IMAGE[str(x)])
        split_size = len(list_file) - int(len(list_file) * SPLIT_SIZE)
        print(split_size)
        train = list_file[:split_size]
        test = list_file[split_size:]
        for file in train:
            out_file = os.path.splitext(file)[0].split('_')
            rfile = out_file[0] + '_' + out_file[1] + '_' + out_file[2] + '_' + str(x) + '.jpg'
            print(rfile)
            img_prepare(IMAGE[str(x)] + file, IMAGE_PATH_STORE_TRAIN + rfile)

        for file in test:
            out_file = os.path.splitext(file)[0].split('_')
            rfile = out_file[0] + '_' + out_file[1] + '_'+ out_file[2] + '_' + str(x) + '.jpg'
            print(rfile)
            img_prepare(IMAGE[str(x)] + file, IMAGE_PATH_STORE_TEST + rfile)

def build_filters():
    ksize = 5
    theta = np.pi*0.25
    lamda = np.pi*0.25
    kern = cv2.getGaborKernel((ksize, ksize), 15.0, theta, lamda, 0.8, 0, ktype=cv2.CV_32F)
    kern /= 1.5*kern.sum()
    return kern


def img_prepare(path_read, path_store):
    img = cv2.imread(path_read, cv2.IMREAD_GRAYSCALE)
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
    mn = np.mean(img)
    if mn > 124:
        k = (mn - 124) * 0.85 / 124 + 0.85
        img = cv2.multiply(img, k)
    if mn < 100:
        k = (mn - 100) * 1.85 / 100 + 1.85
        img = cv2.multiply(img, k)
    img = cv2.medianBlur(img, 7)
    fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
    mn = np.mean(fimg)
    if mn < 80:
        k = (mn - 80) * 1.1 / 80 + 1.1
        fimg = cv2.multiply(fimg, k)
    if mn > 100:
        k = (mn - 100) * 1.1 / 100 + 1.1
        fimg = cv2.multiply(fimg, k)
#    struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#    im_erod = cv2.erode(im_ad, struct_element)
#    im_dilate = cv2.dilate(im_erod, struct_element)
#    ot_tr, im_dilate = cv2.threshold(im_dilate, 40, 150, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#    hi_tr = ot_tr
#    lo_tr = ot_tr * 0.3
#    im_can = cv2.Canny(im_dilate, lo_tr, hi_tr)
    fimg = cv2.adaptiveThreshold(fimg, 10, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 3, 1)
    fimg = cv2.GaussianBlur(fimg, (3, 3), 0)
    cv2.imwrite(path_store, fimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    main()
