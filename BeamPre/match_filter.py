import numpy as np
from array_response import array_response
from velocity_vector import velocity_vector


def match_filter(r, theta, vr, vt, M, N, d, lambda_, Ts, s):
    """
    Matched filter for a specific location and a specific velocity
    Inputs:
        r: distance of the target
        theta: direction of the target
        vr: radial velocity of the target
        vt: transverse velocity of the target
        M: number of antennas at the BS
        N: length of one CPI
        d: antenna spacing at the BS
        lambda_: signal wavelength
        Ts: symbol period
        s: transmit signal
    Outputs:
        X: matched filter
    """
    X = np.zeros((M, N), dtype=complex)
    n = np.arange(1, N + 1)
    d_n = velocity_vector(r, theta, M, d, lambda_, vr, vt, Ts, n)
    a = array_response(r, theta, M, d, lambda_)
    H = a[:, np.newaxis] * d_n
    for n in range(N):
        s_n = s[:, n]
        hn = H[:, n]
        Hn = np.outer(hn, hn.conj())
        X[:, n] = Hn @ s_n
    return X


