import sys
import csv
import numpy as np


def load_data(path):
    return np.array([[float(x) for x in line.split("\n")[0].split("\t")] for line in open(path, "r")])


def np2csv(data, new_path):
    f = open(new_path, "w")
    for one in data:
        f.write(str(one[0]) + "," + str(one[1]) + "," + str(np.random.randint(10 ** 3, 10 ** 9)) + "\n")
    f.close()



if __name__ == "__main__":
    path = sys.argv[1]
    new_path = path.split(".txt")[0] + ".csv"
    print(new_path)
    data = load_data(path)
    np2csv(data, new_path)
