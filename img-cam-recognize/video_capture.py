import cv2
from datetime import datetime
import threading


class VideoCap(object):
    def __int__(self):
        self.capture = False
        self.vpath = ''
        self.img = None
        self.img_time = None
        self.th_break = False

    def __del__(self):
        self.th_break = True

    def start_capture(self, rtsp):
        self.th_break = True
        self.st = threading.Thread(target=self.capture, args=(rtsp))
        self.st.start()

    def capture(self, vpath):
        try:
            cap = cv2.VideoCapture(vpath)
            while not self.th_break:
                if cap.isOpened():
                    ret, frame = cap.read()
                    self.img = frame
                    self.img_time = datetime.now()
                else:
                    cap = cv2.VideoCapture(vpath)
        except Exception as e:
            print(e)

    def image(self):
        return self.img, self.img_time
