

N = 200 * 10 ** 4
from math import sqrt
N_ELEMENTS = int(sqrt(N))
X_min, X_max = 0, N_ELEMENTS
Y_min, Y_max = 0, N_ELEMENTS
W_min, W_max = 1, 10 ** 5

N_SETS = 4 * 10 ** 2
N_PARENTS = int(N_SETS * 0.3) # 175
N_CROSSOVER = N_SETS - N_PARENTS

N_MUTATIONS = 1

ITER = 1 * 10 ** 2
RANDOM_FACTOR = 0.2

