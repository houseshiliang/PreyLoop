import numpy as np
import math


def compute_dp_of_advanced_composition(q,k,sigma,delta):  # sigma σ = noise

    if sigma <= 0:
        print('sigma must be larger than 0.')
    if k <= 0:
        print('k larger than 0.')
    if delta <= 0 or delta >= 1:
        print('delta must be larger than 0 and smaller than 1')
    eps = q*math.sqrt(16*k*np.log(1.25/delta)*np.log(1/delta)/(sigma**2))
    return eps