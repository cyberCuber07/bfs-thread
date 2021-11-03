import numpy as np
from icecream import ic
import os
import csv
from parms import N, X_min, X_max, Y_min, Y_max, W_min, W_max


def one_set(n):
    x = np.zeros((n, 3))
    for i in range(n):
        x[i, 0] = np.random.randint(low=X_min, high=X_max)
        x[i, 1] = np.random.randint(low=Y_min, high=Y_max)
        x[i, 2] = np.random.randint(low=W_min, high=W_max)
    return x


def get_name(save_dir, file_type):
    max_idx = 0
    for _, _, files in os.walk(save_dir):
        for file in files:
            max_idx = max(max_idx, int(file[:len(file_type) - 3]))
    return str(max_idx + 1) + file_type


def save2file(n=N, save_path="data"):
    save_file = os.path.join(save_path, get_name(save_path, ".csv"))
    f = open(save_file, mode="w")
    f_csv = csv.writer(f)
    x = one_set(N)
    for one in x:
        row = [str(i) for i in one]
        f_csv.writerow(row)


if __name__ == "__main__":
    save2file()
