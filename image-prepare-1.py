import os

import cv2

IMAGE = {
         '2': 'Sorted/2/',
         '3': 'Sorted/3/',
         '4': 'Sorted/4/'}
IMAGE_PATH_STORE_TRAIN = 'Imgtrain1/train/'
IMAGE_PATH_STORE_TEST = 'Imgtrain1/test/'
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
    #im = cv2.GaussianBlur(im, (3, 3), 0)
    #im = cv2.adaptiveThreshold(im, 255,
    #                          cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 57, 1)
    cv2.imwrite(path_store, im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    main()
