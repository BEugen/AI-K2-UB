import cv2
from datetime import datetime
import threading
import time


class VideoCap(object):
    def __init__(self):
        self.vpath = ''
        self.img = []
        self.img_time = ''
        self.th_no_break = True
        self.cap = None

    def __del__(self):
        self.th_break = True

    def start_capture(self, rtsp):
        self.vpath = rtsp

    def __capture(self):
        try:
            self.cap = cv2.VideoCapture(self.vpath)
            if self.cap.isOpened():
                print('can read')
                ret, frame = self.cap.read()
                self.img = frame
                self.img_time = datetime.now()
                self.cap.release()
                self.cap = None
        except Exception as e:
            print(e)

    def image(self):
        self.__capture()
        return self.img, self.img_time
