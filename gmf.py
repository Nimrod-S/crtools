from cosmology import *

# All fields are in uG, all distnces are in pc
# TODO citations
B_random = 5
l_random = 60
D_random = 3e3

# Larmor radius in parsecs
#   B in uG, R in V
def Rl(R, B):
    # 1e9 from si/cgs conversion (tesla=volt*second/meter^2), the other number for cm->parsecs
    return 1e9 * R / B / C / 3.086e18

# In radians
def deflection_random(R):
    return np.sqrt(l_random * D_random) / Rl(R, B_random) * np.sqrt(2/9)

