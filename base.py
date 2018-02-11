import psycopg2
import base64
import uuid
from datetime import datetime, timedelta
import cv2, numpy as np


class Psql(object):
    def __init__(self):
        self.connect = psycopg2.connect("dbname='aik2web' user='aik2webusr' host='localhost' password='yjdfzptkfylbz3'")

    def save(self, img, pstop, pempty, pfull):
        cv2.imshow('Sobel p', img)
        buff = base64.b64encode(cv2.imencode('.jpg', img)[1])
        cur = self.connect.cursor()
        guid = uuid.uuid4()
        cur.execute("INSERT INTO conveyer2firstr VALUES(%s, %s, %s, %s, %s, %s)",
                    (str(guid), datetime.now(), float(pstop), float(pempty), float(pfull),
                     buff.decode('utf-8')))
        self.connect.commit()
        cur.close()
        return guid

    def loadlast(self):
        cur = self.connect.cursor()
        cur.execute("SELECT * FROM conveyer2firstr order by stamp desc LIMIT 1")
        data = cur.fetchall()
        cur.close()
        im = base64.b64decode(data[0][5].encode('utf-8'))
        im = cv2.imdecode(np.asarray(bytearray(im), dtype='uint8'), cv2.IMREAD_COLOR)
        return im, data[0][0]

    def updatelast(self, guid):
        cur = self.connect.cursor()
        cur.execute("UPDATE conveyer2firstr SET stamp = %s WHERE  id = %s", (datetime.now(), str(guid)))
        self.connect.commit()
        cur.close()

    def savecategory(self, guid, nclass, predict):
        guid_next = uuid.uuid4()
        cur = self.connect.cursor()
        cur.execute("UPDATE conveyer2firstr SET id-next = %s WHERE  id = %s", (str(guid_next), str(guid)))
        cur.execute("INSERT INTO conveyer2next VALUES(%s, %s, %s, %s, %s, %s, %s)",
                    (str(guid_next), nclass, predict[0], predict[1], predict[2],
                     predict[3], predict[4]))
        self.connect.commit()

    def savestatistic(self, nclass):
        guid = uuid.uuid4()
        cur = self.connect.cursor()
        cur.execute("SELECT * FROM convstat order by start desc LIMIT 1")
        data = cur.fetchall()
        if len(data):
            cur.execute("INSERT INTO convstat VALUES(%s, %s, %s)",
                        (str(guid), nclass, datetime.now()))
        else:
            row = data[0]
            curr_time = datetime.now()
            st_time = row[2]
            if (curr_time - st_time).minute > 480:
                end_time = datetime(st_time.year, st_time.month, st_time.day, int(st_time.hour / 8) * 8 - 1, 59, 59)
                while True:
                    id = uuid.uuid4()
                    cur.execute("UPDATE convstat SET end = %s WHERE  id = %s", (end_time, str(row[0])))
                    cur.execute("INSERT INTO convstat VALUES(%s, %s, %s)",
                                (str(id), row[1], end_time + timedelta(seconds=1)))
                    end_time = end_time + timedelta(hours=8)
                    if end_time > curr_time:
                        if row[1] != nclass:
                            cur.execute("UPDATE convstat SET end = %s WHERE  id = %s",
                                        (curr_time - timedelta(seconds=1), str(id)))
                            cur.execute("INSERT INTO convstat VALUES(%s, %s, %s)",
                                        (str(uuid.uuid4()), nclass, curr_time))
                        break
            else:
                if row[1] != nclass:
                    cur.execute("UPDATE convstat SET end = %s WHERE  id = %s",
                                (curr_time - timedelta(seconds=1), str(row[0])))
                    cur.execute("INSERT INTO convstat VALUES(%s, %s, %s)",
                                (str(uuid.uuid4()), nclass, curr_time))

        cur.close()

    def __del__(self):
        self.connect.close()
