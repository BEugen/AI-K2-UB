
import cv2, numpy as np
import os
from skimage.measure import compare_ssim


IMG_PATH = '/home/eugen/PycharmProjects/AI-K2-UB/Imgtrain2/train/'




def main():
    list_file = os.listdir(IMG_PATH)
    while True:
        list_remove = []
        for x in range(1, len(list_file)):
            imageA = cv2.resize(cv2.imread(IMG_PATH + list_file[0]), (224, 224))
            imageB = cv2.resize(cv2.imread(IMG_PATH + list_file[x]), (224, 224))
            grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            (score, diff) = compare_ssim(grayA, grayB, full=True)
            diff = (diff * 255).astype("uint8")
            if score > 0.7:
                print("SSIM: {}".format(score))
                list_remove.append(list_file[x])
        for f in list_remove:
            list_file.remove(f)
        list_file.remove(list_file[0])
        print(len(list_file))
        if len(list_file) == 0:
            break











if __name__ == '__main__':
    main()
