import numpy as np
def compute_zcdp_random_reshuffling(epoch,sigma):
    if sigma<=0:
        return 0
    zcdp=1/(2*sigma**2)
    return epoch*zcdp

def zcdp_convert_dp(zcdp,delta):
    eps=zcdp + 2*np.sqrt(zcdp * np.log(1/delta))
    return eps

def compute_dp_through_zcdp_random_reshuffling(k,sigma,delta):
    zcdp=compute_zcdp_random_reshuffling(k,sigma)
    eps=zcdp_convert_dp(zcdp,delta)
    return eps