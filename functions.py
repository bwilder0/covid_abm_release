import numpy as np
from numba import jit




@jit(nopython=True,nogil=True)
def categorical_sample(p):
    threshold = np.random.rand()
    current = 0
    for i in range(p.shape[0]):
        current += p[i]
        if current > threshold:
            return i
        
@jit(nopython=True,nogil=True)
#def threshold_exponential(mean):
#    return 1 + np.round(np.random.exponential(mean-1))
def threshold_exponential(mean):
    return np.round(np.random.exponential(mean))

@jit(nopython=True,nogil=True)
def threshold_log_normal(mean, sigma):
    x = np.random.lognormal(mean, sigma)
    if x <= 0:
        return 1
    else:
        return np.round(x)


@jit(nopython=True)
def resevoir_sample(n, k):
    R = np.zeros(k, dtype=np.int32)
    if k == 0:
        return R
    for i in range(k):
        R[i] = i
    W = np.exp(np.log(np.random.rand())/k)
    while i < n:
        i = i + int(np.floor(np.log(np.random.rand())/np.log(1-W))) + 1
        if i < n:
            R[np.random.randint(0, k)] = i
            W = W * np.exp(np.log(np.random.rand())/k)
    return R