from PIL import Image
import os

IMAGE = {'0': '/home/administrator/cam6/Empty/', '1': '/home/administrator/cam6/Material/'}
IMAGE_PATH_STORE = '/home/administrator/cam6/Imgtrain/'
IMAGE_CLASS = '1'

def main():
    for x in IMAGE:
        list_file = os.listdir(IMAGE[x])
        for file in list_file:
            out_file = os.path.splitext(file)[0]
            print(out_file)
            img = Image.open(IMAGE[x] + file)
            img = img.crop((21, 52, 266, 266))
            img.save(IMAGE_PATH_STORE + out_file + '_' + x + '.jpg')


if __name__ == '__main__':
    main()
