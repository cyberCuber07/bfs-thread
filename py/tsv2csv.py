
import sys
import csv
import numpy as np



def load_file(path):
    return np.array([[a for a in line.split("\n")] for line in open(path, "r")])


if __name__ == "__main__":
    path = sys.argv[1]
    data = load_file(path)

    print(data.shape)
