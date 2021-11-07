
import sys
import matplotlib.pyplot as plt
import csv
import numpy as np

path = sys.argv[1]

data = np.array([[int(float(x)) for x in line] for line in csv.reader(open(path, "r"))])

print( data[:, 0] )
print( data[:, 1] )

plt.bar(range(data.shape[0]), data[:, 1])
plt.show()
