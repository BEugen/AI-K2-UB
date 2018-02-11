import os

import cv2

IMAGE_PATH = '/home/eugen/PycharmProjects/AI-K2-UB/img/'
IMAGE_PATH_STORE = '/home/eugen/PycharmProjects/AI-K2-UB/img/imgp/'
IMAGE_CLASS = '1'
SPLIT_SIZE = 0.25


def main():
    list_file = os.listdir(IMAGE_PATH)
    for file in list_file:
        out_file = os.path.splitext(file)[0].split('_')
        print(out_file)
        if out_file[len(out_file) - 1] == IMAGE_CLASS:
            img_prepare(IMAGE_PATH + file, IMAGE_PATH_STORE + out_file[0] + '_' +
                     out_file[1] + '_' + out_file[2] + '.jpg')


def img_prepare(path_read, path_store):
    im = cv2.imread(path_read)
    im = im[200:750, 300:850]
    cv2.imwrite(path_store, im,  [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    main()
