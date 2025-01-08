import numpy as np


def array_response(r, theta, M, d, lambda_):
    """
    Near-field array response vector
    Inputs:
        r: distance of the target
        theta: direction of the target
        M: number of antennas at the BS
        d: antenna spacing at the BS
        lambda_: signal wavelength
    Outputs:
        a: array response vector
    """
    delta_m = np.arange(-(M-1)/2, (M+1)/2) * d
    # propagation distance
    r_m = np.sqrt(r**2 + delta_m**2 - 2 * r * delta_m * np.cos(theta))
    # array response vector
    a = np.exp(-1j * 2 * np.pi / lambda_ * r_m) / r_m
    return a