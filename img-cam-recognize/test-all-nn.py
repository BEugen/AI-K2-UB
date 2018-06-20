import recognize
import cv2, numpy as np
import os
import csv
import math

FILE_DIRECTORY = ('/mnt/data/data/ub-im/AI-K2-UB/Sorted/2',
                  '/mnt/data/data/ub-im/AI-K2-UB/Sorted/3',
                  '/mnt/data/data/ub-im/AI-K2-UB/Sorted/4')
HEADER = ['path', 'iclass', 'snn1', 'snn1s', 'snn2', 'snn2s', 'snn3', 'snn3s']

WEIGHT_RESULT = {'snn1': 1.1, 'snn2': 1.1, 'snn3': 0.9}

def main():
    with open('img.csv', "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(HEADER)
        rc = recognize.RecognizeK2()
        for p in FILE_DIRECTORY:
            files = os.listdir(p)
            for file in files:
                im = cv2.imread(p + '/' + file)
                result = rc.recognize(im)
                csv_row = [file]
                sfile = os.path.splitext(file)[0].split('_')
                try:
                    iclass = int(sfile[len(sfile) - 1])
                except:
                    continue
                if iclass < 2:
                    continue
                csv_row.append(str(iclass))
                for i in range(1, 4):
                    if 'snn' + str(i) in result.keys():
                        text, sm = calc_result(result['snn' + str(i)])
                        csv_row.append(text)
                        csv_row.append(sm)
                writer.writerow(csv_row)

def snn_calc_result(snn):
    rc_snn = []
    print(snn)
    for i in range(0, 4):
        t = []
        for x in range(0, 3):
            rc = int(snn[x][i])
            if rc > 1:
                t.append(float(rc) * WEIGHT_RESULT['snn' + str(x + 1)])
        if len(t) > 0:
            rc_snn.append(np.mean(t))
    rc_calc = math.ceil(np.mean(rc_snn))
    print(rc_calc)
    if rc_calc >= 5:
        return 4
    if rc_calc == 4:
        return 3
    if rc_calc >= 2 and rc_calc <= 3:
        return 2
    return rc_calc


def calc_result(text):
    ln = len(text)
    if ln < 4:
        for _ in range(0, 4 - ln):
            text = text + '1'
    if ln > 4:
        text = text[0:4]
    sm = 0
    for x in range(0, 4):
        sm = sm + int(text[x])
    return text, sm


if __name__ == '__main__':
    t = ['5551', '5232', '3350']
    snn_calc_result(t)
