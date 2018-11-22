import numpy as np
from timeit import default_timer as timer
from numba import vectorize, cuda

# @vectorize(['float32(float32, float32)'], target='cuda')
def vectorAdd(a, b):
    return a * b

def main():
    for _ in range(50):
        N = 32000000
        A = np.ones(N, dtype=np.float32)
        B = np.full((N), 2, dtype=np.float32)
        C = np.zeros(N, dtype=np.float32)

        start = timer()
        C = vectorAdd(A,B)
        vectoradd_time = timer() - start

        print(vectoradd_time)

if __name__ == '__main__':
    main()
