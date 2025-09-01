import numpy as np
import scipy as sp

# --- COSMOLOGY ---
OM = 0.3
H0 = 70         # (km/s)/Mpc
H100 = H0 / 100
C = 299742.448  # km/s

# Assuming flat universe
def sqrtH(z):
    return np.sqrt(OM * (1+z)**3 + 1 - OM)

_zs = np.linspace(0, 1.5, 7500)
_dprop = C / H0 * sp.integrate.cumulative_trapezoid(1/ sqrtH(_zs), _zs, initial=0) # TODO actually this is the comoving distance

z2dprop = sp.interpolate.interp1d(_zs, _dprop)
dprop2z = sp.interpolate.interp1d(_dprop, _zs)
