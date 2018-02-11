from PIL import Image
import os

IMAGE = {'0': 'E:\\opencv-py\\'}
IMAGE_PATH_STORE = 'E:\\opencv-py-p\\'
IMAGE_CLASS = '1'

def main():
    for x in IMAGE:
        list_file = os.listdir(IMAGE[x])
        for file in list_file:
            out_file = os.path.splitext(file)[0]
            print(out_file)
            img = Image.open(IMAGE[x] + file)
            img = img.crop((261, 191, 980 + 261, 766 + 191))
            img.save(IMAGE_PATH_STORE + out_file + '.jpg')


if __name__ == '__main__':
    main()
