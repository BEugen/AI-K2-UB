import cv2, numpy as np
import time
from datetime import datetime
import threading

FILE_PATH = 'D:/opencv-py/'
FRAME_COUNT = 5
RTSP = 'rtsp://admin:123456Qw@172.31.182.138:554/Streaming/channels/102'

def write_img(path, img, fr):
    time_now = datetime.now().strftime('%d-%m-%y %H-%M-%S')
    cv2.imwrite(path + 'CAM30_' + str(fr) + '_' + time_now + '.jpg', img)


def main():
    cap = cv2.VideoCapture(RTSP)
    fr = 0
    fr_ind = FRAME_COUNT
    while True:
        if cap.isOpened():
            ret, frame = cap.read()
            #cv2.imshow('Video', frame)
            if fr_ind == 0:
                st = threading.Thread(target=write_img, args=(FILE_PATH, frame, fr))
                st.start()
                fr_ind = FRAME_COUNT
            fr = fr + 1
            fr_ind = fr_ind - 1
        else:
            cap = cv2.VideoCapture(RTSP)



if __name__ == '__main__':
    main()
