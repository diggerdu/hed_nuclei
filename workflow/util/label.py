import csv
import soundfile as sf
import sounddevice as sd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--csvFile", required=True, help="csv file")
parser.add_argument("-l", "--label", help="label")
args = parser.parse_args()

with open(args.csvFile, mode='r') as f:
    reader = csv.reader(f)
    playList = [row[0] for row in reader if row[1] == args.l or args.l is None]
    print(playList)
