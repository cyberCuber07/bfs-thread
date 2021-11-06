
import sys
import csv
import numpy as np


path = sys.argv[1]

data = np.array([[float(x) for x in line] for line in csv.reader(open(path, "r"))])[:10]
data = data.astype('int')

print(data)
