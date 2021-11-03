
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv



def load_data(path):
    f = csv.reader(open(path, "r"))

    data = np.array([[float(x) for x in line] for line in f])

    return data


def vis(data):
    plt.scatter(data[:, 0], data[:, 1], linewidths=1)
    plt.show()


def main():
    path = sys.argv[1]
    data = load_data(path)
    vis(data)


if __name__ == "__main__":
    main()
