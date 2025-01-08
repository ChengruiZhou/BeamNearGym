import numpy as np


def velocity_vector(r, theta, M, d, lambda_, vr, vt, Ts, n):
    """
    Calculating the near-field Doppler-frequency vector at time index n
    Inputs:
        r: distance of the target
        theta: direction of the target
        M: number of antennas at the BS
        d: antenna spacing at the BS
        lambda_: signal wavelength
        vr: radial velocity of the target
        vt: transverse velocity of the target
        Ts: symbol period
        n: time index
    Outputs:
        d: Doppler-frequency vector
        v_m: velocity vector at all antennas
    """


    delta_m = np.arange(-(M - 1) / 2, (M + 1) / 2) * d
    r_m = np.sqrt(r ** 2 + delta_m ** 2 - 2 * r * delta_m * np.cos(theta))

    vr_m = (r - delta_m * np.cos(theta)) / r_m * vr
    vt_m = delta_m * np.sin(theta) / r_m * vt

    v_m = vr_m + vt_m

    # Doppler frequency vector
    d_n = np.exp(-1j * 2 * np.pi / lambda_ * v_m[:, np.newaxis] * n * Ts)
    return d_n
