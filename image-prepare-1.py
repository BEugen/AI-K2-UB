import os

import cv2

IMAGE = {'0': '/home/eugen/PycharmProjects/AI-K2-UB/Imgtrain3/0/',
         '1': '/home/eugen/PycharmProjects/AI-K2-UB/Imgtrain3/1/',
         '2': '/home/eugen/PycharmProjects/AI-K2-UB/Imgtrain3/2/',
         '3': '/home/eugen/PycharmProjects/AI-K2-UB/Imgtrain3/3/',
         '4': '/home/eugen/PycharmProjects/AI-K2-UB/Imgtrain3/4/'}
IMAGE_PATH_STORE_TRAIN = '/home/eugen/PycharmProjects/AI-K2-UB/Imgtrain4/train/'
IMAGE_PATH_STORE_TEST = '/home/eugen/PycharmProjects/AI-K2-UB/Imgtrain4/test/'
IMAGE_CLASS = '1'
SPLIT_SIZE = 0.25


def main():
    for x in IMAGE:
        list_file = os.listdir(IMAGE[x])
        split_size = int(len(list_file)*SPLIT_SIZE)
        print(split_size)
        train = list_file[:split_size]
        test = list_file[split_size:]
        for file in train:
            out_file = os.path.splitext(file)[0].split('_')
            print(out_file)

            img_prepare(IMAGE[x] + file, IMAGE_PATH_STORE_TRAIN + out_file[0] + '_' +
                         out_file[1] + '_' + out_file[2] + '_' + x + '.jpg')

        for file in test:
            out_file = os.path.splitext(file)[0].split('_')
            print(out_file)
            img_prepare(IMAGE[x] + file, IMAGE_PATH_STORE_TEST + out_file[0] + '_' +
                         out_file[1] + '_' + out_file[2] + '_' + x + '.jpg')


def img_prepare(path_read, path_store):
    im = cv2.imread(path_read, cv2.IMREAD_GRAYSCALE)
    im = cv2.Canny(im, 100, 250)
    cv2.imwrite(path_store, im,  [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    main()
