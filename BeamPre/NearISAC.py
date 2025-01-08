import numpy as np
import matplotlib.pyplot as plt

def db2mag(db):
    return 10 ** (db / 20)

# Parameters
dx = 0.5  # x element spacing
dy = 0.5  # y element spacing
theta0 = 45 * (np.pi / 180)  # Elevation angle
phi0 = 30 * (np.pi / 180)  # Azimuth angle
range_val = 10
lamda = 0.125
alpha_x = 2 * np.pi * np.sin(np.deg2rad(phi0)) * np.cos(np.deg2rad(theta0))  # x phase difference
alpha_y = 2 * np.pi * np.sin(np.deg2rad(theta0))  # y phase difference

M = 20  # Number of elements in x
N = 20  # Number of elements in y

# Element positions
X = np.arange(0, M) * dx  # x array
Y = np.arange(0, N) * dy  # y array
X2 = np.kron(np.ones(N), X)  # Repeat x array for each y
Y2 = np.kron(Y, np.ones(M))  # Repeat y array for each x

# Plotting
plt.figure()
plt.plot(X2, Y2, '.')
plt.axis('equal')
plt.grid(True)
plt.title('Antenna Array')
plt.xlabel('Distance')
plt.ylabel('Distance')
plt.show()



# Steering Vectors
ax = np.exp((complex(0, 1)) * X * alpha_x)
ay = np.exp((complex(0, 1)) * Y * alpha_y)
axy = np.kron(ax, ay)

# dtheta = 0.2
# dphi = 0.2
# theta_scan = np.arange(-90, 90 + dtheta, dtheta)
# phi_scan = np.arange(-90, 90 + dphi, dphi)
# theta_len = len(theta_scan)
# phi_len = len(phi_scan)
# beam = np.zeros((theta_len, phi_len))
#
# for r in range(1, range_val + 1):
#     for j in range(phi_len):
#         theta = 45
#         phi = phi_scan[j]
#         Fx = np.exp((complex(0, 1)) * 2 * np.pi / lamda) * (X * np.sin(theta) * np.cos(phi) - (X ** 2 * (1 - (np.cos(phi) ** 2 * (np.sin(theta) ** 2))) / (2 * r)))
#         Fy = np.exp((complex(0, 1)) * 2 * np.pi / lamda) * (Y * np.cos(phi) - (Y ** 2 * (np.sin(theta) ** 2) / (2 * r)))
#         Fxy = np.kron(Fx, Fy)
#         beam[r-1, j] = np.abs(np.dot(axy.T.conj(), Fxy))
#
# beam_db = 20 * np.log10(beam / np.max(beam))

# CR under CC design
yita = 1 / np.pi
delta = []
p = np.arange(0, 120 + 20, 20)

for i in np.arange(-0.3, 0.5, 0.1):
    for j in np.arange(-0.67, 0.76, 0.1):
        delta_value = (2 / 3) * np.arctan((j * i) / ((np.sin(theta0) * np.cos(phi0)) * np.sqrt((np.sin(theta0) * np.cos(phi0)) ** 2 + j ** 2 + i ** 2))) + ((np.sin(theta0) * np.cos(phi0)) * i * j) / (3 * ((np.sin(theta0) * np.cos(phi0)) ** 2 + i ** 2) * np.sqrt((np.sin(theta0) * np.cos(phi0)) ** 2 + i ** 2 + j ** 2))
        delta.append(delta_value)

delta_sum = np.sum(delta)
CRcc_25 = np.log2(db2mag(p)) + np.log2((1 / (4 * np.pi)) * db2mag(delta_sum))
CRcc_23 = np.log2(1 + db2mag(p) * (1 / (4 * np.pi) * db2mag(delta_sum)))
A = lamda ** 2 / (4 * np.pi)

h_buffer = []
for i in range(-7, 8):
    for j in range(-7, 8):
        r = 10 * np.sqrt((i * 0.00625 - np.sin(np.pi / 4) * np.sin(np.pi / 6)) ** 2 + (j * 0.00625 - np.sqrt(2) / 2) ** 2 + 3 / 8)
        H = np.sqrt(A * ((10 ** 3 * (np.sqrt(6) / 4) ** 3 + (10 ** 2 * (np.sqrt(6) / 4)) ** 2 * (10 * (np.sqrt(2) / 2) - j * (lamda / 2)) ** 2) / (4 * np.pi * r ** 5))) * np.exp(-(complex(0, 1)) * 2 * np.pi * r / lamda)
        h_buffer.append(H)

h_norm = np.linalg.norm(h_buffer) ** 2
CRcc_22 = np.log2(1 + db2mag(p) * h_norm)

h_buffer1 = []
for i in range(-7, 8):
    for j in range(-7, 8):
        r = 5 * np.sqrt((i * 0.0125 - np.sin(np.pi / 4) * np.sin(-np.pi / 6)) ** 2 + (j * 0.0125 - np.sqrt(2) / 2) ** 2 + 3 / 8)
        H = np.sqrt(A * ((5 ** 3 * (np.sqrt(6) / 4) ** 3 + (5 * (np.sqrt(6) / 4)) ** 2 * (5 * (np.sqrt(2) / 2) - j * (lamda / 2)) ** 2) / (4 * np.pi * r ** 5))) * np.exp(-(complex(0, 1)) * 2 * np.pi * r / lamda)
        h_buffer1.append(H)

h_norm1 = np.linalg.norm(h_buffer1) ** 2

# CR under SC design
delta1 = []
for i in np.arange(-0.26, 0.46, 0.01):
    for j in np.arange(-0.7, 0.8, 0.1):
        delta1_value = (2 / 3) * np.arctan((j * i) / ((np.sin(theta0) * np.cos(phi0)) * np.sqrt((np.sin(theta0) * np.cos(phi0)) ** 2 + j ** 2 + i ** 2))) + ((np.sin(theta0) * np.cos(phi0)) * i * j) / (3 * ((np.sin(theta0) * np.cos(phi0)) ** 2 + i ** 2) * np.sqrt((np.sin(theta0) * np.cos(phi0)) ** 2 + i ** 2 + j ** 2))
        delta1.append(delta1_value)

rou = (np.conj(h_norm) * h_norm1) ** 2 / (h_norm * h_norm1)
CRsc_46 = np.log2(1 + (db2mag(p) * rou) * db2mag(np.sum(delta1)))

plt.figure()
plt.plot(p, CRcc_25, linestyle=':', color='r', linewidth=1.5, label='Approximation')
plt.plot(p, CRcc_23, linestyle='-', color='b', linewidth=1.5, marker='o', label='Eq. (23)')
plt.plot(p, CRcc_22, linestyle='-', color='g', linewidth=1.5, marker='*', label='Eq. (22)')
plt.plot(p, CRsc_46, linestyle='-', color='k', linewidth=1.5, marker='s', label='Eq. (46)')
plt.legend()
plt.xlabel('p [dB]')
plt.ylabel('CR bps/Hz')
plt.grid(True)
# plt.show()

# Parameters
p = np.arange(40, 141, 20)
L = 4


# SR under CC design
SRcc_41 = (1 / L) * (np.log2(db2mag(p)) + 2 * np.log2(np.sqrt(L * 5.2143e-04) / (4 * np.pi) * db2mag(np.sum(delta) + 21)))
SRcc_33 = (1 / L) * np.log2(1 + (db2mag(p) * L * rou / (16 * np.pi ** 2)) * db2mag(np.sum(delta) + 21) ** 2)
SRcc_32 = (1 / L) * np.log2(1 + db2mag(p) * L * db2mag(h_norm1) ** 2 * rou)

# SR under SC design
SRsc_39 = (1 / L) * np.log2(1 + db2mag(p) * L * db2mag(h_norm1) ** 2 * 5.2143e-04)
SRcc_40 = (1 / L) * np.log2(1 + 5.2143e-04 * (db2mag(p) * L / (16 * np.pi ** 2)) * db2mag(np.sum(delta1) + 21) ** 2)

# Plotting
plt.figure()
plt.plot(p, SRcc_41, linestyle=':', color='k', linewidth=1.5, label='Approximation')
plt.plot(p, SRcc_33, linestyle='-', color='r', linewidth=1.5, marker='o', label='Eq. (33)')
plt.plot(p, SRsc_39, linestyle='-', color='b', linewidth=1.5, marker='*', label='Eq. (39)')
plt.plot(p, SRcc_32, linestyle='-', linewidth=1.5, marker='d', label='Eq. (32)')
plt.plot(p, SRcc_40, linestyle=':', linewidth=1.5, marker='d', label='Eq. (40)')
plt.legend()
plt.xlabel('p [dB]')
plt.ylabel('SR bps/Hz')
plt.grid(True)
# plt.show()


# CR under SC design
# Calculate rou
rou = (np.conj(h_norm) * h_norm) ** 2 / h_norm ** 2

# CR under SC design
CRsc_46 = np.log2(1 + (db2mag(p) * rou * db2mag(yita / (4 * np.pi))) * db2mag(np.sum(delta)))

# CR-N curve
h2_buffer = []

for N in range(100, 501, 50):
    for i1 in np.arange(round((1 - N) / 2, 1), round((N - 1) / 2, 1), 10):
        for j1 in np.arange(round((1 - N) / 2, 1), round((N - 1) / 2, 1), 10):
            r = 5 * np.sqrt((i1 * 0.0125 - 0.3536) ** 2 + (j1 * 0.0125 - np.sqrt(2) / 2) ** 2 + 3 / 8)
            H1 = np.sqrt(A * ((5 ** 3 * (np.sqrt(6) / 4) ** 3 + (5 * (np.sqrt(6) / 4)) ** 2 * (5 * (np.sqrt(2) / 2) - j1 * (lamda / 2)) ** 2) / (4 * np.pi * r ** 5))) * np.exp(-(complex(0, 1)) * 2 * np.pi * r / lamda)
            h2_buffer.append(H1)

h2_norm_cal = []
for i2 in range(100, 501, 50):
    h2_norm_cal.append(np.linalg.norm(h2_buffer[(i2 - 1) : i2 + 6]))

CRcc_22_inf = np.log2(1 + db2mag(90) * np.array(h2_norm_cal))

plt.figure()
plt.plot(range(100, 501, 50), CRcc_22_inf, linestyle='-', color='r')
plt.xlabel('N')
plt.ylabel('CR bps/Hz')
plt.grid(True)
# plt.show()

# Downlink rate region
R_56_buffer = []
R_57_buffer = []

for tau in np.arange(0, 1.1, 0.1):
    R_56 = (1 / L) * np.log2(1 + db2mag(90) * L * h_norm1 * ((tau ** 2 * rou * h_norm * h_norm1 + (1 - tau) ** 2 * db2mag(h_norm1 ** 2) + 2 * tau * (1 - tau) * db2mag(h_norm) * db2mag(h_norm1 ** 1.5))) / (tau ** 2 * db2mag(h_norm) + (1 - tau) ** 2 * db2mag(h_norm1) + 2 * tau * (1 - tau) * db2mag(h_norm * h_norm1)))
    R_56_buffer.append(R_56)
    R_57 = np.log2(1 + db2mag(90) * ((tau ** 2 * db2mag(h_norm ** 2) + (1 - tau) ** 2 * rou * h_norm1 * h_norm + 2 * tau * (1 - tau) * db2mag(h_norm1) * db2mag(h_norm ** 1.5))) / (tau ** 2 * db2mag(h_norm) + (1 - tau) ** 2 * db2mag(h_norm1) + 2 * tau * (1 - tau) * db2mag(h_norm ** 0.5) * db2mag(h_norm1 ** 0.5)))
    R_57_buffer.append(R_57)

plt.figure()
plt.plot(R_57_buffer, R_56_buffer, linestyle=':', color='g', linewidth=1.5, marker='.')
plt.legend(['Downlink rate region'])
plt.xlabel('CR[bps/Hz]')
plt.ylabel('SR bps/Hz')
plt.axhline(1.76, linestyle='-', color='black')
plt.axvline(15, linestyle='-', color='black')
plt.grid(True)
# plt.show()

#  CC design rc=2r
h3_buffer = []
h4_buffer = []
ha_buffer = []

for ri in range(1, 26):
    for i in np.arange(2.5, 3.1, 1):
        for j in np.arange(-3, 4, 1):
            r = 2 * ri * np.sqrt((i * (0.0625 / (2 * ri)) - 0.3536) ** 2 + (j * (0.0625 / (2 * ri)) - np.sqrt(2) / 2) ** 2 + 3 / 8)
            h_USW = np.sqrt(A / (4 * np.pi * (ri ** 2))) * np.exp(-(complex(0, 1)) * (2 * np.pi / lamda) * r)
            h_NUSW = np.sqrt(A / (4 * np.pi * (r ** 2))) * np.exp(-(complex(0, 1)) * 2 * np.pi * r / lamda)
            h_acc = np.sqrt(A * (((2 * ri) ** 3 * (np.sqrt(6) / 4) ** 3 + ((2 * ri) ** 2 * (np.sqrt(6) / 4)) ** 2 * (2 * ri * (np.sqrt(2) / 2) - j * (lamda / 2)) ** 2) / (4 * np.pi * (2 * ri) ** 5))) * np.exp(-(complex(0, 1)) * 2 * np.pi * r / lamda)
            h4_buffer.append(h_NUSW)
            h3_buffer.append(h_USW)
            ha_buffer.append(h_acc)

h3_norm_cal = [np.linalg.norm(h3_buffer[ii:ii + 7]) for ii in range(0, len(h3_buffer), 7)]
h4_norm_cal = [np.linalg.norm(h4_buffer[ii:ii + 7]) for ii in range(0, len(h4_buffer), 7)]
ha_norm_cal = [np.linalg.norm(ha_buffer[ii:ii + 7]) for ii in range(0, len(ha_buffer), 7)]

CR22_NUSW = np.log2(1 + db2mag(85) * (np.array(h4_norm_cal) ** 2))
CR22_USW = np.log2(1 + db2mag(85) * (np.array(h3_norm_cal) ** 2))
CR22_acc = np.log2(1 + db2mag(85) * (np.array(ha_norm_cal) ** 2))

plt.figure()
plt.plot(range(1, 26), CR22_USW, linestyle=':', linewidth=1.5, label='USW/UPW')
plt.plot(range(1, 26), CR22_NUSW, linestyle='-', linewidth=1.5, label='NUSW')
plt.plot(range(1, 26), CR22_acc, linestyle='-', linewidth=1.5, label='Accurate')
plt.legend()
plt.xlabel('r')
plt.ylabel('CR bps/Hz')
plt.grid(True)
# plt.show()

# SC under r
# SC design rs=r
h5_buffer = []
h6_buffer = []
hacc_buffer=[]
for ri_s in range(2, 51, 2):
    for i in np.arange(2.5, 3, 1):
        for j in range(-3, 4):
            r = ri_s * np.sqrt((i * (0.0625 / ri_s) - 0.3536) ** 2 + (j * (0.0625 / (2 * ri_s)) - np.sqrt(2) / 2) ** 2 + 3 / 8)
            h_USW_s = np.sqrt(A / (4 * np.pi * (ri_s ** 2))) * np.exp(-1j * (2 * np.pi / lamda) * r)
            h_NUSW_s = np.sqrt(A / (4 * np.pi * (r ** 2))) * np.exp(-1j * 2 * np.pi * r / lamda)
            h_acc_s = np.sqrt(A * ((ri_s ** 3 * (np.sqrt(6) / 4) ** 3 + (ri_s ** 2 * (np.sqrt(6) / 4)) ** 2 * (ri_s * (np.sqrt(2) / 2) - j * (lamda / 2)) ** 2) / (4 * np.pi * ri_s ** 5))) * np.exp(-1j * 2 * np.pi * r / lamda)
            h6_buffer.append(h_NUSW_s)
            h5_buffer.append(h_USW_s)
            hacc_buffer.append(h_acc_s)


h5_norm_cal = [np.linalg.norm(h5_buffer[ii:ii + 7]) for ii in range(0, len(h5_buffer), 7)]
h6_norm_cal = [np.linalg.norm(h6_buffer[ii:ii + 7]) for ii in range(0, len(h6_buffer), 7)]
hacc_norm_cal = [np.linalg.norm(hacc_buffer[ii:ii + 7]) for ii in range(0, len(hacc_buffer), 7)]

CR22_NUSW_s = np.log2(1 + db2mag(85) * (np.array(h6_norm_cal) ** 2))
CR22_USW_s = np.log2(1 + db2mag(85) * (np.array(h5_norm_cal) ** 2))
CR22_acc_s = np.log2(1 + db2mag(85) * (np.array(hacc_norm_cal) ** 2))

plt.figure()
plt.plot(range(2, 51, 2), CR22_USW_s, linestyle=':', linewidth=1.5, label='USW/UPW')
plt.plot(range(2, 51, 2), CR22_NUSW_s, linestyle='-', linewidth=1.5, label='NUSW')
plt.plot(range(2, 51, 2), CR22_acc_s, linestyle='-', linewidth=1.5, label='Accurate')
plt.legend()
plt.xlabel('r')
plt.ylabel('SC bps/Hz')
plt.grid(True)
# plt.show()

#####################################################
# uplink CR under SC&CC design

# Parameters
pc = np.arange(25, 106, 5)
ps = 85 # dB

# Compute CRsc_69
CRsc_69 = np.log2(1 + 4 * db2mag(pc) * db2mag(np.sum(delta)) / (16 * np.pi ** 2 + np.sum(delta1) ** 2 * db2mag(ps)))

# Compute CRsc_68
CRsc_68 = np.log2(1 + db2mag(pc) * db2mag(h_norm) / (1 + db2mag(ps) * db2mag(h_norm1) ** 2))

# Compute CRcc_up
CRcc_up = np.log2(1 + db2mag(pc) * h_norm1)

# Compute CRsc_70
CRsc_70 = np.log2(db2mag(pc)) + np.log2(4 * np.pi * np.sum(delta) / (16 * np.pi ** 2 + np.sum(delta1) ** 2 * db2mag(ps)))

# Plotting
plt.figure()
plt.plot(pc, CRsc_68, linestyle=':', color='b', linewidth=1.5, marker='o', label='Eq. (68)')
plt.plot(pc, CRcc_up, linestyle=':', color='r', linewidth=1.5, marker='*', label='uplink-CRcc')
plt.plot(pc, CRsc_69, linestyle=':', color='k', linewidth=1.5, marker='^', label='CR-SC lower bound')
plt.plot(pc, CRsc_70, linestyle=':', color='y', linewidth=1.5, label='high-SNR approximation')
plt.legend()
plt.xlabel('pc [dB] with ps=85dB')
plt.ylabel('CR bps/Hz')
plt.grid(True)
# plt.show()


# uplink SR under CC design
# Parameters
ps = np.arange(45, 126, 5)
pc = 60

# SR calculations
SRcc_63 = (1 / L) * np.log2(1 + (db2mag(ps) * L * db2mag(h_norm1 ** 2) * (h_norm1 - db2mag(pc) * h_norm * h_norm1 / (1 + db2mag(pc) * h_norm))))
SRcc_64 = (1 / L) * np.log2(1 + db2mag(ps) * L * yita ** 2 * np.sum(delta1) ** 2 / (16 * np.pi ** 2 + 4 * np.pi * yita * db2mag(pc) * np.sum(delta)))
SRcc_65 = (1 / L) * (np.log2(db2mag(ps)) + np.log2(L * yita ** 2 * np.sum(delta) ** 2 / (16 * np.pi ** 2 + 4 * np.pi * yita * db2mag(pc) * np.sum(delta))))
SRsc_up = (1 / L) * np.log2(1 + db2mag(ps) * L * yita ** 2 * np.sum(delta1) ** 2 / (16 * np.pi ** 2))

# Plotting
plt.figure()
plt.plot(ps, SRcc_63, linestyle=':', color='r', linewidth=1.5, marker='^', label='Eq. (63)')
plt.plot(ps, SRcc_65, linestyle='-', color='k', linewidth=1.5, marker='o', label='high SNR_Approximation')
plt.plot(ps, SRcc_64, linestyle=':', color='y', linewidth=1.5, label='ISAC CC-lower bound')
plt.plot(ps, SRsc_up, linestyle='-', color='b', linewidth=1.5, marker='*', label='ISAC-SC')
plt.legend()
plt.xlabel('ps [dB] with pc=60dB')
plt.ylabel('SR bps/Hz')
plt.grid(True)

# Channel vectors
ax = np.exp((complex(0, 1)) * X * alpha_x)
ay = np.exp((complex(0, 1)) * Y * alpha_y)
axy = np.kron(ax, ay)

dtheta = 0.2
dphi = 0.2
theta_scan = np.arange(-90, 91, dtheta)
phi_scan = np.arange(-90, 91, dphi)
theta_len = len(theta_scan)
phi_len = len(phi_scan)
beam = np.zeros((theta_len, phi_len))

# Beamforming calculation
for i in range(theta_len):
    for j in range(phi_len):
        theta = theta_scan[i]
        phi = phi_scan[j]
        Fx = np.exp((complex(0, 1)) * X * 2 * np.pi * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)))
        Fy = np.exp((complex(0, 1)) * Y * 2 * np.pi * np.sin(np.deg2rad(phi)))
        Fxy = np.kron(Fx, Fy)
        beam[i, j] = np.abs(np.dot(axy.T.conj(), np.transpose(Fxy)))

beam_db = 20 * np.log10(beam / np.max(beam))

# Plot radiation pattern
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
phi_scan, theta_scan = np.meshgrid(phi_scan, theta_scan)
ax.plot_surface(phi_scan, theta_scan, beam_db, cmap='viridis')
ax.set_title('Radiation pattern')
ax.set_xlabel('Elevation')
ax.set_ylabel('Azimuth')
ax.set_zlabel('Amplitude (dB)')
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_zlim(-80, 10)
# plt.show()

# 2D Plot
plt.figure()
plt.imshow(beam_db, extent=(-90, 90, -90, 90), aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Rectangular surface array orientation diagram')
plt.xlabel('Elevation')
plt.ylabel('Azimuth')
plt.show()

