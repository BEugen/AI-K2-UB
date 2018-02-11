import os
import shutil

IMAGE = {'0': '/home/eugen/PycharmProjects/AI-K2-UB/img-w1/0/',
         '1': '/home/eugen/PycharmProjects/AI-K2-UB/img-w1/1/'
         }
IMAGE_PATH_STORE = '/home/eugen/PycharmProjects/AI-K2-UB/img-w/'
IMAGE_CLASS = '1'
SPLIT_SIZE = 0.25


def main():
    for x in IMAGE:
        list_file = os.listdir(IMAGE[x])
        for file in list_file:
            out_file = os.path.splitext(file)[0].split('_')
            print(out_file)
            shutil.copy2(IMAGE[x] + file, IMAGE_PATH_STORE + out_file[0] + '_' +
                         out_file[1] + '_' + out_file[2] + '_' + x + '.jpg')



if __name__ == '__main__':
    main()
