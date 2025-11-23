import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import scipy as sp
import propagation
from cosmology import *

e0s = np.logspace(19, 20.5)
ceiling = 1e24

zs = np.linspace(0, 0.4, 401)[1:] # HMMMMMMMMMMMMM
zs = np.logspace(-5, -0.5, 401)[1:] # HMMMMMMMMMMMMM

ds = z2dprop(zs) # HMMMMMMMMMMMMMMMM

cumspec_nuc = []
cumspec_pro = []
cumspec_pro2 = []

cumspec_iron = []
cumspec_sil = []
cumspec_cno = []

cumspec_nuc2 = []
cumspec_iron2 = []
cumspec_sil2 = []
cumspec_cno2 = []

mpc_in_km = 3.086e19
model = -2

import tqdm
for e0 in tqdm.tqdm(e0s):
    # cumspec_nuc.append(
    #     sp.integrate.cumulative_trapezoid(
    #         propagation.calc_cosmic_ray_rate_density(e0, ceiling, zs, model, 0),
    #         ds
    #     )[-1] / mpc_in_km ** 2 / (4 * np.pi)
    # )
    cumspec_nuc2.append(
        sp.integrate.cumulative_trapezoid(
            propagation.calc_cosmic_ray_rate_density(e0, ceiling, zs, model),
            ds
        )[-1] / mpc_in_km ** 2 / (4 * np.pi)
    )
    # cumspec_iron.append(
    #     sp.integrate.cumulative_trapezoid(
    #         propagation.calc_cosmic_ray_rate_density(e0, ceiling, zs, model, 0, lowa=38.5, higha=56),
    #         ds
    #     )[-1] / mpc_in_km ** 2 / (4 * np.pi)
    # )
    # cumspec_sil.append(
    #     sp.integrate.cumulative_trapezoid(
    #         propagation.calc_cosmic_ray_rate_density(e0, ceiling, zs, model, 0, lowa=23.5, higha=38.5),
    #         ds
    #     )[-1] / mpc_in_km ** 2 / (4 * np.pi)
    # )
    # cumspec_cno.append(
    #     sp.integrate.cumulative_trapezoid(
    #         propagation.calc_cosmic_ray_rate_density(e0, ceiling, zs, model, 0, lowa=5.5, higha=23.5),
    #         ds
    #     )[-1] / mpc_in_km ** 2 / (4 * np.pi)
    # )
    cumspec_iron2.append(
        sp.integrate.cumulative_trapezoid(
            propagation.calc_cosmic_ray_rate_density(e0, ceiling, zs, model, lowa=38.5, higha=56),
            ds
        )[-1] / mpc_in_km ** 2 / (4 * np.pi)
    )
    cumspec_sil2.append(
        sp.integrate.cumulative_trapezoid(
            propagation.calc_cosmic_ray_rate_density(e0, ceiling, zs, model, lowa=22.5, higha=38.5),
            ds
        )[-1] / mpc_in_km ** 2 / (4 * np.pi)
    )
    cumspec_cno2.append(
        sp.integrate.cumulative_trapezoid(
            propagation.calc_cosmic_ray_rate_density(e0, ceiling, zs, model, lowa=8, higha=22.5),
            ds
        )[-1] / mpc_in_km ** 2 / (4 * np.pi)
    )
    # cumspec_pro.append(
    #     sp.integrate.cumulative_trapezoid(
    #         propagation.calc_cosmic_ray_rate_density(e0, ceiling, zs, 0, 0),
    #         ds
    #     )[-1] / mpc_in_km ** 2 / (4 * np.pi)
    # )
    cumspec_pro2.append(
        sp.integrate.cumulative_trapezoid(
            propagation.calc_cosmic_ray_rate_density_bonus(e0, ceiling + 1e23, zs),
            ds
        )[-1] / mpc_in_km ** 2 / (4 * np.pi)
    )
    # cumspec_pro2.append(1)

    # cumspec_nuc2.append(propagation.integrate_spectrum(e0, -2, 1, 56) / mpc_in_km ** 2 / (4 * np.pi))
    # cumspec_iron.append(propagation.integrate_spectrum(e0, -2, 38.5, 56) / mpc_in_km ** 2 / (4 * np.pi))
    # cumspec_sil.append(propagation.integrate_spectrum(e0, -2, 23.5, 38.5) / mpc_in_km ** 2 / (4 * np.pi))
    # cumspec_cno.append(propagation.integrate_spectrum(e0, -2, 5.5, 23.5) / mpc_in_km ** 2 / (4 * np.pi))
# plt.show()
# spec_nuc = -np.gradient(np.array(cumspec_nuc), e0s)
# spec_pro = -np.gradient(np.array(cumspec_pro), e0s)
spec_pro2 = -np.gradient(np.array(cumspec_pro2), e0s)
spec_nuc2 = -np.gradient(np.array(cumspec_nuc2), e0s)
# spec_iron = -np.gradient(np.array(cumspec_iron), e0s)
# spec_sil = -np.gradient(np.array(cumspec_sil), e0s)
# spec_cno = -np.gradient(np.array(cumspec_cno), e0s)
spec_iron2 = -np.gradient(np.array(cumspec_iron2), e0s)
spec_sil2 = -np.gradient(np.array(cumspec_sil2), e0s)
spec_cno2 = -np.gradient(np.array(cumspec_cno2), e0s)


def smear(e0s, spec, res):
    newspec = []
    # Slow
    for e0 in e0s:
        krn = np.exp(-(e0s - e0)**2/(2 * (e0s * res)**2)) / (np.sqrt(2 * np.pi) * res * e0s)
        newspec.append(sp.integrate.trapezoid(spec * krn, e0s))
    return np.array(newspec)

# plt.xscale("log")
# plt.yscale("log")
def plot_file_spectrum(filepath):
    with open(filepath, "r") as f:
        d = f.read()
    d = d.splitlines()[4:]
    # log10E, E*J, Err_up, Err_low
    # Units: eV, m^-2 s^-1 sr^-1
    data = [tuple(map(float, dd.split())) for dd in d]
    log10e, ej, eup, elow = zip(*data)
    e = 10 ** np.array(log10e)

    # converting to km^-2 yr^-1
    unit_factor = 1e6 * 31556926
    # unit_factor = 1

    full_ej = ej * e * e * unit_factor
    full_elow = elow * e * e * unit_factor
    full_eup = eup * e * e * unit_factor

    plt.errorbar(e, full_ej, [full_elow, full_eup], fmt='o', color='grey', label='Auger spectrum 2019')
    plt.xlim(1e18, e[-1] * 2)
    plt.ylim(min(full_ej) / 5, max(full_ej) * 5)
    return

# plot_file_spectrum("auger_2019.txt")

# np.save("prosp", spec_pro2)

# plt.plot(e0s, spec_nuc * e0s**3, color='brown')
# plt.plot(e0s, spec_pro * e0s**3)
plt.plot(np.log10(e0s), np.log10(spec_pro2 * e0s**3), color="red", linestyle=":", label="proton model", linewidth=3)
plt.plot(np.log10(e0s), np.log10(spec_nuc2 * e0s**3), color='black', linestyle=':', label="nuclei model (total)", linewidth=3)
plt.plot(np.log10(e0s), np.log10(smear(e0s ,spec_nuc2, .07) * e0s**3), color='gray', linestyle=':', label="nuclei model (total)", linewidth=1)
# plt.plot(e0s, spec_iron * e0s**3, color='blue')
# plt.plot(e0s, spec_sil * e0s**3, color='red')
# plt.plot(e0s, spec_cno * e0s**3, color='green')
plt.plot(np.log10(e0s), np.log10(spec_iron2 * e0s**3), color='purple', linestyle=':', label="$A_o>38$", linewidth=3)
plt.plot(np.log10(e0s), np.log10(spec_sil2 * e0s**3), color='blue', linestyle=':', label="$A_o=24-38$", linewidth=3)
plt.plot(np.log10(e0s), np.log10(spec_cno2 * e0s**3), color='green', linestyle=':', label="$A_o=5-23$", linewidth=3)

# plt.grid(True, which="both", ls=":", color='0.65')

plt.xlabel(r"$\text{log}_{10}(E/\text{eV})$")
plt.ylabel(r"$\text{log}_{10}(E^3\Phi(E)$ /($\text{yr}^{-1}$ $\text{km}^{-2} \text{sr}^{-1}$ $\text{eV}^2$))")


plt.xlim((18.033, 20.297))
plt.ylim((35.700, 38.115))
plt.imshow(mpimg.imread("../dumb.png"), extent=[18.033, 20.297, 35.700, 38.115])

# plt.legend()
plt.show()


plt.plot([1,2,3],[1,2,3])
plt.show()

