import sys
import csv


path = sys.argv[1]
new_path = path.split(".")[0] + "-new.csv"

lines = [[float(x) for x in line] for line in csv.reader(open(path, "r"))]

f = open(new_path, "w")
for one in lines[:10**6]:
    f.write(str(one[0]) + "," + str(one[1]) + "," + str(one[2]) + "\n")

f.close()
