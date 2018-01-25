import csv
from collections import Counter,defaultdict
from getPatch import getPatch

patchList = getPatch('../labeling')

def interSetPatch(fileList, excludeList):
    cnt = Counter()
    for fn in fileList:
        with open(fn, 'r') as f:
            csvReader = csv.reader(f)
            for row in csvReader:
                try:
                    assert row[0].endswith('.wav')
                except:
                    continue

                if row[1] in excludeList:
                    continue
                cnt[tuple(row)] += 1
            f.close()

    interSetDict = {k[0]:k[1] for k, v in cnt.items() if v == len(fileList) and not v in patchList}
    print(len(interSetDict))
    with open("unverified.csv", 'w') as f:
        for k, v in interSetDict.items():
            print('{},{}'.format(k, v), file=f)
        f.close

if __name__ == '__main__':
    interSetPatch(['../submission/76.csv', '../submission/gcp.csv'], ['silence', 'unknown'])

