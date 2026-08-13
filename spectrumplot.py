import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tqdm

import numpy as np
import scipy as sp
import propagation
from cosmology import *

e0s = np.logspace(19, 20.5)
ceiling = 1e24

zs = np.linspace(0, 0.4, 401)[1:]
ds = z2dprop(zs)

cumspec_pro = []
cumspec_nuc = []
cumspec_iron = []
cumspec_sil = []
cumspec_cno = []

mpc_in_km = 3.086e19
model = -2

def smear(e0s, spec, res):
    newspec = []
    # Slow
    for e0 in e0s:
        krn = np.exp(-(e0s - e0)**2/(2 * (e0s * res)**2)) / (np.sqrt(2 * np.pi) * res * e0s)
        newspec.append(sp.integrate.trapezoid(spec * krn, e0s))
    return np.array(newspec)


for e0 in tqdm.tqdm(e0s):
    cumspec_nuc.append(
        sp.integrate.cumulative_trapezoid(
            propagation.calc_cosmic_ray_rate_density(e0, ceiling, zs, model), # lowa = 0?
            ds
        )[-1] / mpc_in_km ** 2 / (4 * np.pi)
    )
    cumspec_iron.append(
        sp.integrate.cumulative_trapezoid(
            propagation.calc_cosmic_ray_rate_density(e0, ceiling, zs, model, lowa=38, higha=56), #38.5-56
            ds
        )[-1] / mpc_in_km ** 2 / (4 * np.pi)
    )
    cumspec_sil.append(
        sp.integrate.cumulative_trapezoid(
            propagation.calc_cosmic_ray_rate_density(e0, ceiling, zs, model, lowa=22, higha=38), # 23.5-38.5
            ds
        )[-1] / mpc_in_km ** 2 / (4 * np.pi)
    )
    cumspec_cno.append(
        sp.integrate.cumulative_trapezoid(
            propagation.calc_cosmic_ray_rate_density(e0, ceiling, zs, model, lowa=8, higha=22), # 5.5-23.5
            ds
        )[-1] / mpc_in_km ** 2 / (4 * np.pi)
    )
    cumspec_pro.append(
        sp.integrate.cumulative_trapezoid(
            propagation.calc_cosmic_ray_rate_density_protons(e0, ceiling, zs),
            ds
        )[-1] / mpc_in_km ** 2 / (4 * np.pi)
    )
    

plt.figure(figsize=(5,5))


# plt.show()
spec_pro = -np.gradient(np.array(cumspec_pro), e0s)
spec_nuc = -np.gradient(np.array(cumspec_nuc), e0s)
spec_iron = -np.gradient(np.array(cumspec_iron), e0s)
spec_sil = -np.gradient(np.array(cumspec_sil), e0s)
spec_cno = -np.gradient(np.array(cumspec_cno), e0s)


plt.plot(np.log10(e0s), np.log10(spec_pro * e0s**3), color="red", linestyle=":", label="proton model", linewidth=3)
plt.plot(np.log10(e0s), np.log10(spec_nuc * e0s**3), color='black', linestyle=':', label="nuclei model (total)", linewidth=3)
# plt.plot(np.log10(e0s), np.log10(smear(e0s ,spec_nuc, .07) * e0s**3), color='gray', linestyle=':', label="nuclei model (total)", linewidth=1)
plt.plot(np.log10(e0s), np.log10(spec_iron * e0s**3), color='purple', linestyle=':', label="$A_o>38$", linewidth=3)
plt.plot(np.log10(e0s), np.log10(spec_sil * e0s**3), color='blue', linestyle=':', label="$A_o=24-38$", linewidth=3)
plt.plot(np.log10(e0s), np.log10(spec_cno * e0s**3), color='green', linestyle=':', label="$A_o=5-23$", linewidth=3)

# plt.grid(True, which="both", ls=":", color='0.65')

plt.xlabel(r"$\text{log}_{10}(E/\text{eV})$")
plt.ylabel(r"$\text{log}_{10}(E^3\Phi(E)$ /($\text{yr}^{-1}$ $\text{km}^{-2} \text{sr}^{-1}$ $\text{eV}^2$))")

plt.xlim((18.033, 20.297))
plt.ylim((35.700, 38.115))
plt.imshow(mpimg.imread("../auger_spectrum.png"), extent=[18.033, 20.297, 35.700, 38.115])

# plt.legend()
plt.show()
