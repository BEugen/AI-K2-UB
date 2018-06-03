import recognize
import cv2, numpy as np
import os
import csv

FILE_DIRECTORY = ('/mnt/data/data/ub-im/AI-K2-UB/Sorted/2',
                  '/mnt/data/data/ub-im/AI-K2-UB/Sorted/3',
                  '/mnt/data/data/ub-im/AI-K2-UB/Sorted/4')
HEADER = ['path', 'iclass', 'snn1', 'snn1s', 'snn2', 'snn2s', 'snn3', 'snn3s']


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
    main()
