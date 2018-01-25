import csv
import os
from collections import defaultdict
table = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']

def convertLabel(label):
    if label == '\\':
        return 'silence'
    if label == '\'':
        return 'unknown'
    if label not in table:
        return 'unknown'
    return label

def getPatch(Path):
    patchList = list()
    for fn in os.listdir(Path):
        try:
            assert fn.endswith("csv")
        except:
            continue
        with open(Path+'/'+fn) as f:
            reader = csv.reader(f)
            patchList.extend([row[0] for row in reader])
    return patchList

def getLabelDict(Path):
    labelDict = dict()
    for fn in os.listdir(Path):
        try:
            assert fn.endswith("csv")
        except:
            continue
        with open(Path+'/'+fn) as f:
            reader = csv.reader(f)
            labelDict.update({row[0]:convertLabel(row[1]) for row in reader if row[1] != '#'})
    return labelDict







