import numpy as np
import matplotlib.pyplot as plt
from array_response import array_response
from velocity_vector import velocity_vector
from loss_function import loss_function

# System parameters
Delta = 2e-3  # coherence time (s)
c = 3e8  # speed of light (m/s)
f = 28e9  # carrier frequency (Hz)
lambda_ = c / f  # signal wavelength (m)
M = 256  # number of antennas
d = lambda_ / 2  # antenna spacing (m)
B = 100e3  # system bandwidth (Hz)
Ts = 1 / B  # symbol duration (s)
D = M * d  # array aperture (m)
N = int(np.floor(Delta / Ts))  # coherence period interval
Pt = 10 ** (-10 / 10)  # transmit power (mW)
N0 = 10 ** (-174 / 10) * B  # noise power (mW)
SNR = Pt * lambda_**2 / ((4 * np.pi)**3 * N0)  # signal-to-noise ratio

# Target parameters
r = 10  # distance (m)
theta = np.pi / 2  # direction (rad) (90 degrees)
vr = 10  # radial velocity (m/s)
vt = 8  # transverse velocity (m/s)

# Signal model
a = array_response(r, theta, M, d, lambda_)
w = np.conj(a) / np.sqrt(M)  # beamformer
s = np.sqrt(SNR) * w[:, np.newaxis] / np.sqrt(2) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))  # transmit signal

# Receive signal
A = np.outer(a, a.conj())
Y = np.zeros((M, N), dtype=complex)
for n in range(N):
    s_n = s[:, n]
    d_n = velocity_vector(r, theta, M, d, lambda_, vr, vt, Ts, n+1)  # MATLAB indices start at 1
    D_n = np.outer(d_n, d_n.conj())

    z_n = 1 / np.sqrt(2) * (np.random.randn(M) + 1j * np.random.randn(M))
    Y[:, n] = (A * D_n) @ s_n + z_n

# Plot transverse velocity
vt_all = np.arange(-20, 20.5, 0.5)
t_velocity = np.zeros(len(vt_all))
for i in range(len(vt_all)):
    t_velocity[i] = -loss_function(Y, r, theta, [vr, vt_all[i]], M, N, d, lambda_, Ts, s)[0]

t_velocity /= np.max(t_velocity)
plt.figure()
plt.plot(vt_all, t_velocity, linewidth=1.5)
plt.axvline(x=vt, color='m', linestyle='--', linewidth=1)
plt.xlabel('Transverse velocity (m/s)')
plt.ylabel('Normalized $g(\\mathbf{Y}, \\eta, v)$')
plt.grid(True)
plt.xlim(-20, 20)
plt.ylim(0, 1)
plt.show()

# Plot radial velocity
vr_all = np.arange(-20, 20.5, 0.5)
r_velocity = np.zeros(len(vr_all))
for i in range(len(vr_all)):
    r_velocity[i] = -loss_function(Y, r, theta, [vr_all[i], vt], M, N, d, lambda_, Ts, s)[0]

r_velocity /= np.max(r_velocity)
plt.figure()
plt.plot(vr_all, r_velocity, linewidth=1.5)
plt.axvline(x=vr, color='m', linestyle='--', linewidth=1)
plt.xlabel('Radial velocity (m/s)')
plt.ylabel('Normalized $g(\\mathbf{Y}, \\eta, v)$')
plt.grid(True)
plt.show()
