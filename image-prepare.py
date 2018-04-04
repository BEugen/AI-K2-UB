from PIL import Image
import os

IMAGE = {'0': 'E:\\opencv-py-p\\0\\', '1': 'E:\\opencv-py-p\\1\\', '2': 'E:\\opencv-py-p\\2\\'}
IMAGE_PATH_STORE = 'E:\\opencv-py-p\\img\\'
IMAGE_CLASS = '1'

def main():
    for x in IMAGE:
        list_file = os.listdir(IMAGE[x])
        ind = 0
        for file in list_file:
            out_file = os.path.splitext(file)[0]
            print(out_file)
            img = Image.open(IMAGE[x] + file)
           # img = img.crop((21, 52, 266, 266))
            img.save(IMAGE_PATH_STORE + 'CAM_' + str(ind) + '_' + x + '.jpg')
            ind = ind + 1


if __name__ == '__main__':
    main()
