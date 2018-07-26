import psycopg2
import base64
import uuid
from datetime import datetime, timedelta, timezone
import cv2, numpy as np
import pytz


class Psql(object):
    def __init__(self):
        self.connect = psycopg2.connect("dbname='aik2web' user='aik2webusr' host='localhost' password='yjdfzptkfylbz3'")

    def save(self, img, pstop, pempty, pfull, stamp_file):
        buff = base64.b64encode(cv2.imencode('.jpg', img)[1])
        cur = self.connect.cursor()
        guid = uuid.uuid4()
        cur.execute("INSERT INTO aik2_conveyer2firstr VALUES(%s, %s, %s, %s, %s, %s, null, %s)",
                    (str(guid), datetime.now(), float(pstop), float(pempty), float(pfull),
                     buff.decode('utf-8'), stamp_file))
        self.connect.commit()
        cur.close()
        return guid

    def loadlast(self):
        try:
            cur = self.connect.cursor()
            cur.execute("SELECT * FROM aik2_conveyer2firstr order by stamp desc LIMIT 1")
            data = cur.fetchall()
            cur.close()
            if len(data) == 0:
                return None, None
            im = base64.b64decode(data[0][5].encode('utf-8'))
            fd = data[0][7]
            im = cv2.imdecode(np.asarray(bytearray(im), dtype='uint8'), cv2.IMREAD_COLOR)
            return im, data[0][0], fd
        except Exception as e:
            print(e)

    def updatelast(self, guid, score):
        cur = self.connect.cursor()
        cur.execute("UPDATE aik2_conveyer2firstr SET stamp = %s, predstop = %s WHERE  id = %s",
                    (datetime.now(), score, str(guid)))
        self.connect.commit()
        cur.close()

    def savecategory(self, guid, nclass, predict):
        guid_next = uuid.uuid4()
        cur = self.connect.cursor()
        cur.execute("UPDATE aik2_conveyer2firstr SET idnext = %s WHERE  id = %s", (str(guid_next), str(guid)))
        cur.execute("INSERT INTO aik2_conveyer2next VALUES(%s, %s, %s, %s, %s, %s, %s)",
                    (str(guid_next), nclass, float(predict[0]), float(predict[1]), float(predict[2]),
                     float(predict[3]), float(predict[4])))
        self.connect.commit()

    def savecategory1(self, guid, nclass, predict):
        guid_next = uuid.uuid4()
        cur = self.connect.cursor()
        cur.execute("UPDATE aik2_conveyer2firstr SET idnext = %s WHERE  id = %s", (str(guid_next), str(guid)))
        cur.execute("INSERT INTO aik2_conveyer2next VALUES(%s, %s, %s, %s, %s, %s, %s)",
                    (str(guid_next), nclass, float(predict[0]), float(predict[1]), -1.0, -1.0, -1.0))
        self.connect.commit()

    def savecategory2(self, guid, nclass, predict):
        guid_next = uuid.uuid4()
        cur = self.connect.cursor()
        cur.execute("UPDATE aik2_conveyer2firstr SET idnext = %s WHERE  id = %s", (str(guid_next), str(guid)))
        cur.execute("INSERT INTO aik2_conveyer2next VALUES(%s, %s, %s, %s, %s, %s, %s)",
                    (str(guid_next), nclass, -1.0, -1.0, float(predict[0]),
                     float(predict[1]), float(predict[2])))
        self.connect.commit()

    def savestatistic(self, nclass):
        guid = uuid.uuid4()
        cur = self.connect.cursor()
        cur.execute("SELECT * FROM aik2_convstat order by start desc LIMIT 1")
        data = cur.fetchall()
        cur.execute("SELECT * FROM aik2_conv2seconds where aik2_conv2seconds.nclass = %s "
                    "order by aik2_conv2seconds.ndate desc LIMIT 1;", (nclass,))
        srow = cur.fetchall()
        if len(data) == 0:
            cur.execute("INSERT INTO aik2_convstat VALUES(%s, %s, %s)",
                        (str(guid), nclass, datetime.now()))
        else:
            row = data[0]
            curr_time = datetime.now().replace(tzinfo=None)
            st_time = row[2].replace(tzinfo=None)
            delta_time = ((curr_time - timedelta(seconds=1)) - st_time).total_seconds()
            sec_dtime = datetime(curr_time.year, curr_time.month, curr_time.day, 0, 0, 0, tzinfo=pytz.UTC)
            prev_dtime = srow[0][2].replace(tzinfo=None)
            prev_dtime = datetime(prev_dtime.year, prev_dtime.month, prev_dtime.day, 0, 0, 0, tzinfo=pytz.UTC)
            if row[1] != nclass:
                if srow and len(srow) > 0 and sec_dtime == prev_dtime:
                    rsrow = srow[0]
                    cur.execute("UPDATE aik2_conv2seconds SET seconds = %s WHERE  id = %s",
                                (delta_time + rsrow[3], str(rsrow[0])))
                else:
                    cur.execute("INSERT INTO aik2_conv2seconds VALUES(%s, %s, %s, %s)",
                                (str(uuid.uuid4()), nclass, sec_dtime, delta_time))

                cur.execute("UPDATE aik2_convstat SET \"end\" = %s WHERE  id = %s",
                            (curr_time - timedelta(seconds=1), str(row[0])))
                cur.execute("INSERT INTO aik2_convstat VALUES(%s, %s, %s)",
                            (str(uuid.uuid4()), nclass, curr_time))
            if curr_time.hour != st_time.hour and row[1] == nclass:
                cm_time = datetime(st_time.year, st_time.month, st_time.day, st_time.hour, 59, 59, tzinfo=pytz.UTC)
                cur.execute("UPDATE aik2_convstat SET \"end\" = %s WHERE  id = %s",
                            (cm_time, str(row[0])))
                cur.execute("INSERT INTO aik2_convstat VALUES(%s, %s, %s)",
                            (str(uuid.uuid4()), nclass, cm_time + timedelta(seconds=1)))
                delta_time = (cm_time - st_time).total_seconds()
                if srow and len(srow) > 0 and sec_dtime == prev_dtime:
                    nsrow = srow[0]
                    cur.execute("UPDATE aik2_conv2seconds SET seconds = %s WHERE  id = %s",
                                delta_time + nsrow[3], str(nsrow[0]))
                else:
                    cur.execute("INSERT INTO aik2_conv2seconds VALUES(%s, %s, %s, %s)",
                                (str(uuid.uuid4()), nclass, sec_dtime, delta_time))
        self.connect.commit()
        cur.close()

    def getcropimg(self):
        try:
            cur = self.connect.cursor()
            cur.execute("SELECT * FROM aik2_conveyer2imgcrop order by tstamp desc LIMIT 1")
            data = cur.fetchall()
            cur.close()
            if len(data) == 0:
                return None, None, None, None
            return data[0][2], data[0][3], data[0][4], data[0][5]
        except Exception as e:
            print(e)
            return None

    def savedata(self, img, data, tfile):
        buff = base64.b64encode(cv2.imencode('.jpg', img)[1])
        cur = self.connect.cursor()
        guid = uuid.uuid4()

        cur.execute("INSERT INTO aik2_conveyer2status VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (str(guid), datetime.now(),  float(data['fnn']['0']),
                     float(data['fnn']['1']), float(data['fnn']['2']), data['snn1'], data['snn2'], data['snn3'],
                     float(data['snn']),
                     buff.decode('utf-8'), tfile, float(data['sck'])))
        self.connect.commit()
        cur.close()

    def loadimglast(self):
        try:
            cur = self.connect.cursor()
            cur.execute("SELECT * FROM aik2_conveyer2status order by tstamp desc LIMIT 1")
            data = cur.fetchall()
            cur.close()
            if len(data) == 0:
                return None
            im = base64.b64decode(data[0][9].encode('utf-8'))
            im = cv2.imdecode(np.asarray(bytearray(im), dtype='uint8'), cv2.IMREAD_COLOR)
            return im, data[0][0], data[0][10] #img, guid, time img file
        except Exception as e:
            print(e)

    def updateimglast(self, guid, stop):
        cur = self.connect.cursor()
        cur.execute("UPDATE aik2_conveyer2status SET tstamp = %s, stop = %s WHERE  id = %s",
                    (datetime.now(), stop, str(guid)))
        self.connect.commit()
        cur.close()

    def __del__(self):
        self.connect.close()
