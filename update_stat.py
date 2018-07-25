import psycopg2
import base64
import uuid
from datetime import datetime, timedelta
import cv2, numpy as np
import pytz



def get_json_thrend():
    try:
        connect = psycopg2.connect("dbname='aik2web' user='aik2webusr' host='localhost' password='yjdfzptkfylbz3'")
        dtc = datetime.now()
        dte = datetime(dtc.year, dtc.month, dtc.day, dtc.hour, dtc.minute, dtc.second, tzinfo=pytz.UTC)
        dte = dte - timedelta(days=1)
        dtc = datetime(dtc.year, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        while dtc < dte:
            for y in range(0, 6):
                cur = connect.cursor()
                cur.execute("SELECT * FROM aik2_convstat WHERE start >= %s and \"end\" < %s  "
                           "and nclass = %s and \"end\" is not null",
                           (dtc, (dtc + timedelta(days=1)), (y - 1)))
                sql_val = cur.fetchall()
                delta = 0
                if sql_val and len(sql_val) > 0:
                    for r in sql_val:
                        delta += (r[3] - r[2]).total_seconds()
                print((y-1), delta)
                cur.execute("INSERT INTO aik2_conv2seconds VALUES(%s, %s, %s, %s)",
                            (str(uuid.uuid4()), (y-1), dtc, delta))
                connect.commit()
            dtc = dtc + timedelta(days=1)

    except Exception as e:
        print(e)
        return []

get_json_thrend()