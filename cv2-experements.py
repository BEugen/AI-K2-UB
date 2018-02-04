import cv2, numpy as np
import os


FILE_P = '/home/eugen/PycharmProjects/AI-K2-UB/cv2img/CAM6_19012018_19:20_99.99_1.jpg'

def main():
    im_p = cv2.imread(FILE_P, cv2.IMREAD_GRAYSCALE)
    sobel_p = cv2.Sobel(im_p, cv2.CV_64F, 0, 1, ksize=3)
    canny_p = cv2.Canny(im_p, 100, 250)
    cv2.imshow('Input image', im_p)
    cv2.imshow('Sobel p', sobel_p)
    cv2.imshow('Sobel p', canny_p)

    cv2.waitKey()


if __name__ == '__main__':
    main()