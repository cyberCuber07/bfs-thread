
import pandas as pd
import sys
import numpy as np


path = sys.argv[1]

data = pd.read_csv(path, header=None).to_numpy()
data = data[1:, :]
data[:, :2] -= 1
data = data.astype('int')
np.savetxt(path.split(".")[0] + "-new.csv", data, fmt='%d', delimiter=',')
