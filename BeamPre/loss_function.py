from operator import truediv

import numpy as np
from setuptools.command.easy_install import easy_install

from array_response import array_response
from match_filter import match_filter
from velocity_vector import velocity_vector


def loss_function(Y, r, theta, v, M, N, d, lambda_, Ts, s):
    """
    Loss function and gradient for maximum likelihood estimation
    Inputs:
        Y: signal received over one CPI
        r: distance of the target
        theta: direction of the target
        v: vector of velocities
        M: number of antennas at the BS
        N: length of one CPI
        d: antenna spacing at the BS
        lambda_: signal wavelength
        Ts: symbol period
        s: transmit signal
        gradient_required: boolean flag to indicate if gradient is required
    Outputs:
        f: loss function
        g: gradient (if gradient_required is True)
    """


    vr = v[0]
    vt = v[1]

    delta_m = np.arange(-(M - 1) / 2, (M + 1) / 2) * d
    r_m = np.sqrt(r ** 2 + delta_m ** 2 - 2 * r * delta_m * np.cos(theta))
    X = match_filter(r, theta, vr, vt, M, N, d, lambda_, Ts, s)

    # Loss function calculation
    f = -np.abs(np.trace(np.conj(X.T) @ Y)) ** 2 / np.real(np.trace(X @ np.conj(X.T)))

    # Gradient calculation (optional)
    g = None

        
    if True:  # Assuming we always need gradient
        n = np.arange(1, N + 1)
        d_n = velocity_vector(r, theta, M, d, lambda_, vr, vt, Ts, n)
        qm = (r - delta_m * np.cos(theta)) / r_m
        pm = delta_m * np.sin(theta) / r_m

        d_vr = -1j * 2 * np.pi / lambda_ * Ts * np.outer(qm, n) * d_n
        d_vt = -1j * 2 * np.pi / lambda_ * Ts * np.outer(pm, n) * d_n
        a = array_response(r, theta, M, d, lambda_)

        H = a[:, np.newaxis] * d_n
        H_vr = a[:, np.newaxis] * d_vr
        H_vt = a[:, np.newaxis] * d_vt

        X_vr = np.zeros((M, N), dtype=complex)
        X_vt = np.zeros((M, N), dtype=complex)
        for n in range(N):
            hn = H[:, n]
            hn_vr = H_vr[:, n]
            hn_vt = H_vt[:, n]
            X_vr[:, n] = (hn_vr @ hn.T + hn @ hn_vr.T) * s[:, n]
            X_vt[:, n] = (hn_vt @ hn.T + hn @ hn_vt.T) * s[:, n]

        X_norm = np.linalg.norm(X, 'fro') ** 2
        Theta = np.trace(Y @ np.conj(X.T)) * X_norm
        Omega = np.abs(np.trace(Y @ np.conj(X.T))) ** 2
        g_X = (Theta * np.conj(Y.T) - Omega * np.conj(X.T)) / X_norm ** 2

        g = np.zeros(2)
        g[0] = -2 * np.real(np.trace(g_X @ X_vr))
        g[1] = -2 * np.real(np.trace(g_X @ X_vt))

    return f, g
