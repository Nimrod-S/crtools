from cosmology import *

import healpy as hp

# All fields are in uG, all distnces are in pc
B_random = 5
l_random = 60
D_random = 3e3

# Larmor radius in parsecs
#   B in uG, R in V
def Rl(R, B):
    # 1e9 from si/cgs conversion (tesla=volt*second/meter^2), the other number for cm->parsecs
    return 1e9 * R / B / C / 3.086e18

# In radians
def deflection_random_old(R):
    return np.sqrt(l_random * D_random) / Rl(R, B_random) * np.sqrt(2/9)

def rmstd(lat):
    return np.abs(7 / np.sin(lat * np.pi/180))

# In cm^-3
nmean = 0.01
# R in V, RMSTD in rad/m^2
def deflection_random(R, RMSTD):
    # units of c are cm^-1
    c = 0.28 / (R/40e18)
    # Convert RMSTD to rad/cm^2
    th = c * 1e-4 * RMSTD / nmean
    return th / np.sqrt(3)

# Take a healpy map of deflection std in each pixel.
# Create a matrix M so that, for a map V, MV is the convolved map. If we think of V as a column vector, each column of M is a single gaussian centered on a pixel.
def defmap_to_matrix(df):
    npix = len(df)
    nside = hp.npix2nside(npix)
    total = []
    for i in range(npix):
        tst = np.zeros(npix)
        tst[i] = 1
        total.append(hp.smoothing(tst, sigma=df[i]))
    return np.array(total)

# 3.748403213 = RIGIDITY