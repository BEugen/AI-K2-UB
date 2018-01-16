from PIL import Image
import os

IMAGE_PATH = '/home/administrator/cam6/Empty/'
IMAGE_PATH_STORE = '/home/administrator/cam6/Empty_Crop/'
IMAGE_CLASS = '1'

def main():
    list_file = os.listdir(IMAGE_PATH)
    for file in list_file:
        out_file = os.path.splitext(file)[0]
        print(out_file)
        img = Image.open(IMAGE_PATH + file)
        img = img.crop((21, 52, 266, 266))
        img.save(IMAGE_PATH_STORE + out_file + '_' + IMAGE_CLASS + '.jpg')


if __name__ == '__main__':
    main()
