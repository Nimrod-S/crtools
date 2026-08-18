import argparse

import numpy as np
import scipy as sp
import healpy as hp
import matplotlib.pyplot as plt

import propagation
import analysis
import mfp
from cosmology import *

def delta(m):
    mm = m/np.mean(m)
    return mm
def forg(g, k, exp):
    if g == -1:
        protons = np.load(f"/mnt/x/uhecr/cr_output/meanmaps/{exp}/mean_v2_m0_s-2_b1.npy")
        nuclei = np.load(f"/mnt/x/uhecr/cr_output/meanmaps/{exp}/mean_v2_m-2_s-2_b1.npy")
    else:
        protons = np.load(f"/mnt/x/uhecr/cr_output/meanmaps/{exp}/mean_v2_U{g}_m0_s-2_b1.npy")
        nuclei = np.load(f"/mnt/x/uhecr/cr_output/meanmaps/{exp}/mean_v2_U{g}_m-2_s-2_b1.npy")
    protonsr = np.sum(protons[2:], axis=0)
    nucleir = np.sum(nuclei[2:], axis=0)
    base = delta(nucleir)
    # if g == -1:
    #     hp.mollview(delta(nucleir), min=min(base), max=max(base), title=f"Nuclei", sub=(2,3,1))
    #     hp.mollview(delta(protonsr), min=min(base), max=max(base), title=f"Protons", sub=(2,3,2))
    # else:
    hp.mollview(delta(nucleir), min=min(base), max=max(base), sub=(2,3,k), title={-1:"No GMF", 0: "base", 6:"twistX"}[g], cbar=False)
    hp.mollview(delta(protonsr), min=min(base), max=max(base), sub=(2,3,k+3), title="", notext=True)

def get_results(args, name, exp, bias, model, sdens):
    d = args.croutput_directory + f"/results/{exp}/"

    fname = f"{name}_b{bias}_m{model}_s{sdens}.npy"

    return np.load(d + fname)


def plot_contours(e0s, a, color):
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"A={a}")
    for i, e0 in enumerate(e0s):
        e1s = np.geomspace(e0, 5e20)
        gs = e1s / propagation.MP / a
        d = propagation.d_analytic(a, gs, e0)
        d = z2dprop(propagation.deff2z(d))
        d2 = propagation.d_semi(a, gs, e0)
        d2 = z2dprop(propagation.deff2z(d2))
        plt.plot(e1s, d2, color=color, alpha=1-.7*i/len(e0s), label=f"{int(e0/1e18)} EeV")
        # plt.plot(e1s, d2, color="darkgreen", alpha=1-.7*i/len(e0s))
        # plt.text(e1s[3], d[3], f"{int(e0/1e18)}", color=color, alpha=1-.6*i/len(e0s), backgroundcolor='white')

def plot_side_contours(a, color):
    threshd = np.load(f"contours{a}.npy")
    
    e1s = np.geomspace(2e19, 2e20)

    for i, thresh in enumerate(threshd):
        plt.plot(e1s, thresh, color=color, alpha=.3+.7*i/len(threshd))


def fig_mfp_comparisons():    
    plt.subplot(221)

    plt.ylabel("$\lambda$ (Mpc)")
    DP.graph_mfp_by_a(56)
    plt.subplot(222)

    
    DP.graph_mfp_by_a(28)
    plt.subplot(223)
    plt.ylabel("$\lambda$ (Mpc)")
    plt.xlabel("$\gamma$")
    DP.graph_mfp_by_a(16)
    # DP.graph_mfp_by_a(14)
    plt.subplot(224)
    plt.xlabel("$\gamma$")
    DP.graph_mfp_by_a(12)
    plt.show()
    return

def fig_effective_distance():
    zs = np.linspace(0, 0.1)
    dprop = z2dprop(zs)
    dlum = dprop * (1+zs)**2
    dhub = zs * C / H0
    deff = propagation.z2deff(zs)
    
    plt.xlabel("$z$")
    plt.ylabel("distance (Mpc)")
    plt.plot(zs, dhub, color='gray', linestyle='--', label="$cz/H_0$")
    plt.plot(zs, dprop, color='black', label="$d_C$")
    plt.plot(zs, dlum, color='blue', label="$d_L$")
    plt.plot(zs, deff, color='goldenrod', label="$\\tilde d$")
    plt.grid(True)
    plt.legend()
    plt.show()
    return

def fig_contours():

    e0s = [2e19, 3e19, 4e19, 5e19, 6e19, 7e19, 8e19, 9e19, 1e20]

    plt.subplot(221)
    plt.ylabel("distance (Mpc)")
    plot_contours(e0s, 56, "darkblue")
    plot_side_contours(56, "darkred")

    plt.subplot(222)
    plot_contours(e0s, 28, "darkblue")
    plot_side_contours(28, "darkred")

    plt.subplot(223)
    plt.xlabel("$E_s$ (eV)")
    plt.ylabel("distance (Mpc)")
    plot_contours(e0s, 16, "darkblue")
    plot_side_contours(16, "darkred")

    plt.subplot(224)
    plt.xlabel("$E_s$ (eV)")
    plot_contours(e0s, 14, "darkblue")
    plot_side_contours(14, "darkred")

    plt.legend()
    plt.show()
    return

def fig_rigidity():
    r = np.logspace(17.5, 19.4)
    r = np.linspace(10 ** 17.5, 10 ** 19.4, 3000)
    r2 = propagation.get_r_dist(3e19, r, -2)
    r4 = propagation.get_r_dist(4e19, r, -2)
    r6 = propagation.get_r_dist(6e19, r, -2)
    # r2 -= r4
    # r4 -= r6
    r2 /= np.sum(r2 * np.gradient(r))
    r2m = np.sum(r2 * r * np.gradient(r))
    r4 /= np.sum(r4 * np.gradient(r))
    r4m = np.sum(r4 * r * np.gradient(r))
    r6 /= np.sum(r6 * np.gradient(r))
    r6m = np.sum(r6 * r * np.gradient(r))

    r2cum = np.cumsum(r2 * np.gradient(r))
    i = np.searchsorted(r2cum, 0.1)
    print(r[i])
    print(r[i+1])
    print(r2m)
    print(r2m / r4m)

    plt.figure()
    plt.plot(r / 1e18, r2, color='black', linestyle='-', label='20 EeV')
    plt.plot(r / 1e18, r4, color='black', linestyle='--', label='40 EeV')
    plt.plot(r / 1e18, r6, color='black', linestyle=':', label='60 EeV')
    plt.xlabel("Rigidity (EV)")
    # plt.vlines([r2m/1e18, r4m/1e18, r6m/1e18], 0, max(r4), linestyles=["-", "-.", ":"], color='brown')
    plt.xlim(r[0] / 1e18, 15)
    plt.legend()
    plt.show()
    return

def fig_distances():
    zs = np.linspace(0, 0.2)

    dndr = propagation.calc_cosmic_ray_rate_density(2e19, 1e21, zs, -2)
    dndrp = propagation.calc_cosmic_ray_rate_density(2e19, 1e21, zs, 0)
    plt.subplot(131)
    plt.title(r"$E_o>20~\text{EeV}$")
    plt.xlim(0, 800)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("distance (Mpc)")
    plt.ylabel(r"$\psi$ (rays $\text{yr}^{-1}$ $\text{Mpc}^{-3})$")
    plt.plot(z2dprop(zs), dndr, color='C0')
    plt.plot(z2dprop(zs), dndrp, color='C3')

    dndr = propagation.calc_cosmic_ray_rate_density(4e19, 1e21, zs, -2)
    dndrp = propagation.calc_cosmic_ray_rate_density(4e19, 1e21, zs, 0)
    plt.subplot(132)
    plt.title(r"$E_o>40~\text{EeV}$")
    plt.xlim(0, 600)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("distance (Mpc)")
    plt.plot(z2dprop(zs), dndr, color='C0')
    plt.plot(z2dprop(zs), dndrp, color='C3')

    dndr = propagation.calc_cosmic_ray_rate_density(6e19, 1e21, zs, -2)
    dndrp = propagation.calc_cosmic_ray_rate_density(6e19, 1e21, zs, 0)
    plt.subplot(133)
    plt.title(r"$E_o>60~\text{EeV}$")
    plt.xlim(0, 300)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("distance (Mpc)")
    plt.plot(z2dprop(zs), dndr, color='C0', label='nuclei')
    plt.plot(z2dprop(zs), dndrp, color='C3', label='protons')

    plt.legend()
    plt.show()
    return

def fig_gmfbins():
    npix = hp.nside2npix(32)
    m = np.ones(npix) * hp.UNSEEN

    l, b = hp.pix2ang(32, np.arange(npix), lonlat=True)
    mlat = 10
    reg0 = np.where(b > 45)
    reg1 = np.where((b <= 45) & (l < 180) & (l > 100) & (np.abs(b) > mlat))
    reg2 = np.where((b <= 45) & (l <= 100) & (l >= 0) & (np.abs(b) > mlat))
    reg3 = np.where((b <= 45) & (l <= 360) & (l >= 270) & (np.abs(b) > mlat))
    reg4 = np.where((b <= 45) & (l < 270) & (l >= 180) & (np.abs(b) > mlat))
    regions = [reg0, reg1, reg2, reg3, reg4]
    
    v = 1
    for r in regions:
        for ipix in r:
            m[ipix] = v
        v += 1
    
    hp.mollview(m, cbar=False, title="Division of the sky to regions (galactic coordinates)", cmap="magma")
    plt.show()

    return

def fig_general_coefficient():
    p = np.load(f"/mnt/x/uhecr/cr_output/meanmaps/auger/mean_v2_m0_s-2_b1.npy")
    n = np.load(f"/mnt/x/uhecr/cr_output/meanmaps/auger/mean_v2_m-2_s-2_b1.npy")
    print(np.sum(p[4:]), np.sum(n[4:]))
    for m in range(8):
        p = np.load(f"/mnt/x/uhecr/cr_output/meanmaps/auger/mean_v2_U{m}_m0_s-2_b1.npy")
        n = np.load(f"/mnt/x/uhecr/cr_output/meanmaps/auger/mean_v2_U{m}_m-2_s-2_b1.npy")
        print(np.sum(p[4:]), np.sum(n[4:]))
    
def fig_largescale():
    plt.figure(figsize=(16, 7))
    forg(-1, 1, "auger")
    forg(0, 2, "auger")
    forg(6, 3, "auger")
    plt.figure(figsize=(16, 7))
    forg(-1, 1, "isotropic")
    forg(0, 2, "isotropic")
    forg(6, 3, "isotropic")
    plt.show()
    return


def get_results_large_vs(args, exp, bias, model, sdens, gmf, mask, calibration):
    magname = "" if (gmf == -1) else f"_U{gmf}"
    # if model == -2:
    #     top = get_results(args, f"TESTmfsnuc_k{mask}_{calibration}{magname}", exp, bias, model, sdens)
    #     bot = get_results(args, f"mfspro_k{mask}_{calibration}{magname}", exp, bias, model, sdens)
    #     return top - bot
    top = get_results(args, f"mfnuc_k{mask}_{calibration}{magname}", exp, bias, model, sdens)
    bot = get_results(args, f"mfpro_k{mask}_{calibration}{magname}", exp, bias, model, sdens)
    return top - bot

def get_results_small_vs(args, exp, bias, model, sdens, gmf, mask, calibration):
    magname = "" if (gmf == -1) else f"_U{gmf}"
    # if model == -2:
    #     top = get_results(args, f"TESTmfsnuc_k{mask}_{calibration}{magname}", exp, bias, model, sdens)
    #     bot = get_results(args, f"mfspro_k{mask}_{calibration}{magname}", exp, bias, model, sdens)
    #     return top - bot
    top = get_results(args, f"ttnuc_k{mask}_{calibration}{magname}", exp, bias, model, sdens)
    bot = get_results(args, f"ttpro_k{mask}_{calibration}{magname}", exp, bias, model, sdens)
    return top - bot

def get_results_small(args, exp, bias, model, sdens, gmf, mask, calibration):
    magname = "" if (gmf == -1) else f"_U{gmf}"
    return get_results(args, f"en16.7_k{mask}_{calibration}{magname}", exp, bias, model, sdens)
    # The line below is for effective entropy
    return get_results(args, f"ens16.7_k{mask}_{calibration}{magname}", exp, bias, model, sdens)

def get_results_energy(args, exp, bias, model, sdens, gmf, mask, calibration):
    magname = "" if (gmf == -1) else f"_U{gmf}"
    return get_results(args, f"ec16.7_k{mask}_{calibration}{magname}", exp, bias, model, sdens)

def get_results_dipole(args, exp, bias, model, sdens, gmf, calibration):
    # These names are using the old convention, in the new convention it's just "mpd_" without the angle.
    magname = "" if (gmf == -1) else f"_U{gmf}"
    if model == -2 and calibration == "e2e42":
        return get_results(args, f"ds16.7_{calibration}{magname}", exp, bias, model, sdens) # TYPO BY ME WHEN CREATING THESE
    return get_results(args, f"mpd16.7_{calibration}{magname}", exp, bias, model, sdens)

def fig_large_mag(args, exp):

    ecal = "mid"
    mask = 10

    plt.figure(figsize=(15, 5))

    for sp in [1, 2, 3]:
        dens = {1:-2, 2:-4, 3:-5}[sp]

        nuc = get_results_large_vs(args, exp, 1, -2, dens, -1, mask, ecal)
        pro = get_results_large_vs(args, exp, 1, 0, dens, -1, mask, ecal)

        bns = np.linspace(min(pro), max(nuc))

        plt.subplot(130+sp)
        plt.title(r"$s_0=10^{"+str(dens)+r"} \text{Mpc}^{-3}$")

        plt.hist(nuc, density=True, alpha=.25, bins=bns, color='C0', label='nuclei')
        plt.hist(pro, density=True, alpha=.25, bins=bns, color='C3', label='protons')

        print(f"HHs{dens}")

        for m in range(8):
            nucm = get_results_large_vs(args, exp, 1, -2, dens, m, mask, ecal)
            prom = get_results_large_vs(args, exp, 1, 0, dens, m, mask, ecal)
            print(np.sum(nucm < -0.016210843886379056) / len(nucm))
    
            plt.hist(nucm, density=True, bins=bns, color='C0', alpha=1-.1*m, histtype='step', linewidth=2)
            plt.hist(prom, density=True, bins=bns,  color='C3', alpha=1-.1*m, histtype='step', linewidth=2)

        if exp == "auger":
            value = 0.1306416531573985 - 0.14685249704377756
            print(np.sum(nuc < value) / len(nuc))
            print(np.sum(pro < value) / len(pro))
            plt.vlines(value, 0, 50, color='black', linestyle='--')

        if dens == -2:
            plt.ylabel("p.d.f")
        plt.xlabel("$T$")
        if dens == -4:
            plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()

def fig_large_cal(args, exp):
    mask = 10
    gmf = 0

    plt.figure(figsize=(15, 5))

    for sp in [1, 2, 3]:
        dens = {1:-2, 2:-4, 3:-5}[sp]

        nuc = get_results_large_vs(args, exp, 1, -2, dens, gmf, mask, "mid")
        pro = get_results_large_vs(args, exp, 1, 0, dens, gmf, mask, "mid")
        nuchigh = get_results_large_vs(args, exp, 1, -2, dens, gmf, mask, "high")
        nuclow = get_results_large_vs(args, exp, 1, -2, dens, gmf, mask, "low")
        prohigh = get_results_large_vs(args, exp, 1, 0, dens, gmf, mask, "high")
        prolow = get_results_large_vs(args, exp, 1, 0, dens, gmf, mask, "low")

        bns = np.linspace(min(prolow), max(nuchigh))

        plt.subplot(130+sp)
        plt.title(r"$s_0=10^{"+str(dens)+r"} \text{Mpc}^{-3}$")

        plt.hist(nuc, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $E>32~$EeV')
        plt.hist(pro, density=True, alpha=.25, bins=bns, color='C3', label='protons, $E>32~$EeV')
        plt.hist(nuchigh, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $E>36.5~$EeV')
        plt.hist(prohigh, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $E>36.5~$EeV')
        plt.hist(nuclow, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, linestyle=":", label='nuclei, $E>27.5~$EeV')
        plt.hist(prolow, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, linestyle=":", label='protons, $E>27.5~$EeV')

        if exp == "auger":
            value = 0.1306416531573985 - 0.14685249704377756
            plt.vlines(value, 0, 50, color='black', linestyle='--')

        if dens == -2:
            plt.ylabel("p.d.f")
        plt.xlabel("$T$")
        if dens == -4:
            plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()

def fig_large_cal2(args, exp):
    mask = 10
    gmf = -1

    plt.figure(figsize=(15, 5))
    dens = -4
    for sp in [1, 2, 3]:
        # dens = {1:-2, 2:-4, 3:-5}[sp]
        cal = ["low", "mid", "high"][sp-1]

        pro = get_results_small_vs(args, exp, 1, 0, dens, gmf, mask, cal)
        prohigh = get_results_small_vs(args, exp, 1.7, 0, dens, gmf, mask, cal)
        prolow = get_results_small_vs(args, exp, 0, 0, dens, gmf, mask, cal)

        bns = np.linspace(min(min(prohigh), min(prolow)), max(max(prohigh), max(prolow)))

        plt.subplot(130+sp)
        # plt.title(r"$s_0=10^{"+str(dens)+r"} \text{Mpc}^{-3}$")
        plt.title(["$E>27.5~$EeV", "$E>32~$EeV", "$E>36.5~$EeV"][sp-1])

        plt.hist(pro, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
        plt.hist(prohigh, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')
        plt.hist(prolow, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, linestyle=":", label='protons, $b_1=0$')

        if exp == "auger":
            value = -0.039475593896871664 # TEMP
            plt.vlines(value, 0, 50, color='black', linestyle='--')

        if dens == -2:
            plt.ylabel("p.d.f")
        plt.xlabel("$T$")
        if dens == -4:
            plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()



def fig_large_results(args, exp):
    mask = 10
    ecal = "mid"
    gmf = -1

    plt.figure(figsize=(15, 5))

    for sp in [1, 2, 3]:
        dens = {1:-2, 2:-4, 3:-5}[sp]

        nuc = get_results_large_vs(args, exp, 1, -2, dens, gmf, mask, ecal)
        pro = get_results_large_vs(args, exp, 1, 0, dens, gmf, mask, ecal)
        nuchigh = get_results_large_vs(args, exp, 1.7, -2, dens, gmf, mask, ecal)
        nuclow = get_results_large_vs(args, exp, 0, -2, dens, gmf, mask, ecal)
        prohigh = get_results_large_vs(args, exp, 1.7, 0, dens, gmf, mask, ecal)
        prolow = get_results_large_vs(args, exp, 0, 0, dens, gmf, mask, ecal)

        bns = np.linspace(min(prolow), max(nuchigh))

        plt.subplot(130+sp)
        plt.title(r"$s_0=10^{"+str(dens)+r"} \text{Mpc}^{-3}$")

        
        plt.hist(nuc, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
        plt.hist(pro, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
        plt.hist(nuchigh, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
        plt.hist(prohigh, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')
        plt.hist(nuclow, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, linestyle=":", label='nuclei, $b_1=0$')
        plt.hist(prolow, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, linestyle=":", label='protons, $b_1=0$')

        if exp == "auger":
            value = 0.1306416531573985 - 0.14685249704377756
            plt.vlines(value, 0, 50, color='black', linestyle='--')
            x = 0.0012
            print(np.sum(prohigh > value) / len(prohigh))
            print(np.sum(prohigh > x) / len(prohigh))
            print(np.sum(pro < x) / len(pro))
            print((np.mean(prohigh) - np.mean(pro))/np.std(pro))
            print(np.sum(pro > value) / len(pro))


        plt.ylabel("p.d.f")
        plt.xlabel("$T$")
        plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    # plt.show()


def fig_small_mag(args, exp):
    ecal = "mid"
    mask = 10

    plt.figure(figsize=(15, 5))

    for sp in [1, 2, 3]:
        dens = {1:-2, 2:-4, 3:-5}[sp]

        nuc = get_results_small(args, exp, 1, -2, dens, -1, mask, ecal)
        pro = get_results_small(args, exp, 1, 0, dens, -1, mask, ecal)

        bns = np.linspace(max(min(nuc), 8.4), max(pro))

        print(f"HHs{dens}")
        print(np.sum(nuc > 8.979925653934787) / len(nuc))

        plt.subplot(130+sp)
        plt.title(r"$s_0=10^{"+str(dens)+r"} \text{Mpc}^{-3}$")

        plt.hist(nuc, density=True, alpha=.25, bins=bns, color='C0', label='nuclei')
        plt.hist(pro, density=True, alpha=.25, bins=bns, color='C3', label='protons')

        for m in range(8):
            nucm = get_results_small(args, exp, 1, -2, dens, m, mask, ecal)
            prom = get_results_small(args, exp, 1, 0, dens, m, mask, ecal)

            print(np.sum(nucm > 8.979925653934787) / len(nucm))
    
            plt.hist(nucm, density=True, bins=bns, color='C0', alpha=1-.1*m, histtype='step', linewidth=2)
            plt.hist(prom, density=True, bins=bns,  color='C3', alpha=1-.1*m, histtype='step', linewidth=2)


        if exp == "auger":
            if mask == 0:
                value = 9.16281363829213
            elif mask == 10:
                value = 8.979925653934787
            plt.vlines(value, 0, 50, color='black', linestyle='--')

        if dens == -2:
            plt.ylabel("p.d.f")
        plt.xlabel("$S$")
        if dens == -4:
            plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()

def fig_small_cal(args, exp):
    mask = 10
    gmf = 0

    plt.figure(figsize=(15, 5))

    for sp in [1, 2, 3]:
        dens = {1:-2, 2:-4, 3:-5}[sp]

        nuc = get_results_small(args, exp, 1, -2, dens, gmf, mask, "mid")
        pro = get_results_small(args, exp, 1, 0, dens, gmf, mask, "mid")
        nuchigh = get_results_small(args, exp, 1, -2, dens, gmf, mask, "high")
        nuclow = get_results_small(args, exp, 1, -2, dens, gmf, mask, "low")
        prohigh = get_results_small(args, exp, 1, 0, dens, gmf, mask, "high")
        prolow = get_results_small(args, exp, 1, 0, dens, gmf, mask, "low")

        bns = np.linspace(max(min(nuchigh), 8.5), max(prolow))

        plt.subplot(130+sp)
        plt.title(r"$s_0=10^{"+str(dens)+r"} \text{Mpc}^{-3}$")

        plt.hist(nuc, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $E>32~$EeV')
        plt.hist(pro, density=True, alpha=.25, bins=bns, color='C3', label='protons, $E>32~$EeV')
        plt.hist(nuchigh, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $E>36.5~$EeV')
        plt.hist(prohigh, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $E>36.5~$EeV')
        plt.hist(nuclow, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, linestyle=":", label='nuclei, $E>27.5~$EeV')
        plt.hist(prolow, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, linestyle=":", label='protons, $E>27.5~$EeV')

        if exp == "auger":
            if mask == 0:
                value = 9.16281363829213
            elif mask == 10:
                value = 8.979925653934787
            plt.vlines(value, 0, 50, color='black', linestyle='--')

        if dens == -2:
            plt.ylabel("p.d.f")
        plt.xlabel("$S$")
        if dens == -4:
            plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()
    
def fig_small_results(args, exp):
    mask = 10
    ecal = "mid"
    gmf = -1

    plt.figure(figsize=(15, 5))

    for sp in [1, 2, 3]:
        dens = {1:-2, 2:-4, 3:-5}[sp]

        nuc = get_results_small(args, exp, 1, -2, dens, gmf, mask, ecal)
        pro = get_results_small(args, exp, 1, 0, dens, gmf, mask, ecal)
        nuchigh = get_results_small(args, exp, 1.7, -2, dens, gmf, mask, ecal)
        nuclow = get_results_small(args, exp, 0, -2, dens, gmf, mask, ecal)
        prohigh = get_results_small(args, exp, 1.7, 0, dens, gmf, mask, ecal)
        prolow = get_results_small(args, exp, 0, 0, dens, gmf, mask, ecal)

        bns = np.linspace(min(nuchigh), max(prolow))

        plt.subplot(130+sp)
        plt.title(r"$s_0=10^{"+str(dens)+r"} \text{Mpc}^{-3}$")

        plt.hist(nuc, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
        plt.hist(pro, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
        plt.hist(nuchigh, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
        plt.hist(prohigh, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')
        plt.hist(nuclow, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, linestyle=":", label='nuclei, $b_1=0$')
        plt.hist(prolow, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, linestyle=":", label='protons, $b_1=0$')

        if exp == "auger":
            if mask == 0:
                value = 9.16281363829213
            elif mask == 10:
                value = 8.979925653934787
            plt.vlines(value, 0, 50, color='black', linestyle='--')

        if dens == -2:
            plt.ylabel("p.d.f")
        plt.xlabel("$S$")
        if dens == -4:
            plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()

def fig_energy_mag(args, exp):

    ecal = "mid"
    mask = 10

    plt.figure(figsize=(15, 5))

    for sp in [1, 2, 3]:
        dens = {1:-2, 2:-4, 3:-5}[sp]

        nuc = get_results_energy(args, exp, 1, -2, dens, -1, mask, ecal)
        pro = get_results_energy(args, exp, 1, 0, dens, -1, mask, ecal)

        bns = np.linspace(min(min(pro), min(nuc)), max(nuc))
        if dens == -4:
            bns = np.linspace(min(min(pro), min(nuc)), 1.6e-4)
        if dens == -5:
            bns = np.linspace(min(min(pro), min(nuc)), 3e-4)

        plt.subplot(130+sp)
        plt.title(r"$s_0=10^{"+str(dens)+r"} \text{Mpc}^{-3}$")

        plt.hist(nuc, density=True, alpha=.25, bins=bns, color='C0', label='nuclei')
        plt.hist(pro, density=True, alpha=.25, bins=bns, color='C3', label='protons')

        print(dens)
        for m in range(8):
            nucm = get_results_energy(args, exp, 1, -2, dens, m, mask, ecal)
            prom = get_results_energy(args, exp, 1, 0, dens, m, mask, ecal)

            print(np.sum(nucm < 0.00014063331503275217) / len(nucm))
            print(np.sum(prom > 0.00014063331503275217) / len(prom))
    
            plt.hist(nucm, density=True, bins=bns, color='C0', alpha=1-.1*m, histtype='step', linewidth=2)
            plt.hist(prom, density=True, bins=bns,  color='C3', alpha=1-.1*m, histtype='step', linewidth=2)


        if exp == "auger":
            if mask == 0:
                value = 0.00011766813765115812
            elif mask == 10:
                value = 0.0001403110718268533
            plt.vlines(value, 0, 1.5e5, color='black', linestyle='--')

        if dens == -2:
            plt.ylabel("p.d.f")
        plt.xlabel("$C_{32,42}$")
        if dens == -4:
            plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()

def fig_energy_cal(args, exp):
    mask = 10
    gmf = 0

    plt.figure(figsize=(15, 5))

    for sp in [1, 2, 3]:
        dens = {1:-2, 2:-4, 3:-5}[sp]

        nuc = get_results_energy(args, exp, 1, -2, dens, gmf, mask, "mid")
        pro = get_results_energy(args, exp, 1, 0, dens, gmf, mask, "mid")
        nuchigh = get_results_energy(args, exp, 1, -2, dens, gmf, mask, "high")
        nuclow = get_results_energy(args, exp, 1, -2, dens, gmf, mask, "low")
        prohigh = get_results_energy(args, exp, 1, 0, dens, gmf, mask, "high")
        prolow = get_results_energy(args, exp, 1, 0, dens, gmf, mask, "low")

        bns = np.linspace(min(prolow), max(nuchigh))

        plt.subplot(130+sp)
        plt.title(r"$s_0=10^{"+str(dens)+r"} \text{Mpc}^{-3}$")

        plt.hist(nuc, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $E>32~$EeV')
        plt.hist(pro, density=True, alpha=.25, bins=bns, color='C3', label='protons, $E>32~$EeV')
        plt.hist(nuchigh, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $E>36.5~$EeV')
        plt.hist(prohigh, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $E>36.5~$EeV')
        plt.hist(nuclow, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, linestyle=":", label='nuclei, $E>27.5~$EeV')
        plt.hist(prolow, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, linestyle=":", label='protons, $E>27.5~$EeV')

        if exp == "auger":
            if mask == 0:
                value = 0.00011766813765115812
            elif mask == 10:
                value = 0.0001403110718268533
            plt.vlines(value, 0, 1.5e5, color='black', linestyle='--')

        if dens == -2:
            plt.ylabel("p.d.f")
        plt.xlabel("$C_{32,42}$")
        if dens == -4:
            plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()

def fig_energy_results(args, exp):
    mask = 10
    ecal = "mid"
    gmf = -1

    plt.figure(figsize=(15, 5))

    for sp in [1, 2, 3]:
        dens = {1:-2, 2:-4, 3:-5}[sp]

        nuc = get_results_energy(args, exp, 1, -2, dens, gmf, mask, ecal)
        pro = get_results_energy(args, exp, 1, 0, dens, gmf, mask, ecal)
        nuchigh = get_results_energy(args, exp, 1.7, -2, dens, gmf, mask, ecal)
        nuclow = get_results_energy(args, exp, 0, -2, dens, gmf, mask, ecal)
        prohigh = get_results_energy(args, exp, 1.7, 0, dens, gmf, mask, ecal)
        prolow = get_results_energy(args, exp, 0, 0, dens, gmf, mask, ecal)

        bns = np.linspace(min(prolow), min(max(nuchigh), 3e-4))

        plt.subplot(130+sp)
        plt.title(r"$s_0=10^{"+str(dens)+r"} \text{Mpc}^{-3}$")

        plt.hist(nuc, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
        plt.hist(pro, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
        plt.hist(nuchigh, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
        plt.hist(prohigh, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')
        plt.hist(nuclow, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, linestyle=":", label='nuclei, $b_1=0$')
        plt.hist(prolow, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, linestyle=":", label='protons, $b_1=0$')

        if exp == "auger":
            if mask == 0:
                value = 0.00011766813765115812
            elif mask == 10:
                value = 0.0001403110718268533
            plt.vlines(value, 0, 1.5e5, color='black', linestyle='--')

        if dens == -2:
            plt.ylabel("p.d.f")
        plt.xlabel("$C_{32,42}$")
        if dens == -4:
            plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- method 1
def fisher_pvalue(vecs, test_vec):
    """Fit a Fisher distribution (isotropic Gaussian-analog on the sphere)
    and return P(angular distance from mean >= that of test_vec)."""
    mean = vecs.mean(axis=0)
    rbar = np.linalg.norm(mean)                  # mean resultant length
    mu = mean / rbar                             # barycenter direction
    kappa = rbar * (3 - rbar**2) / (1 - rbar**2)  # concentration estimate
    cos_t = np.clip(np.dot(test_vec, mu), -1.0, 1.0)
    # Tail of the offset-angle distribution: (e^{k cos t} - e^{-k}) / (e^k - e^{-k})
    p = np.expm1(kappa * (cos_t + 1.0)) / np.expm1(2.0 * kappa)
    return p, mu, kappa
 
 
# ---------------------------------------------------------------- method 2
def empirical_pvalue(vecs, test_vec):
    """Fraction of data points farther from the barycenter than the test point.
    No distributional assumption, but resolution is limited to ~1/N."""
    mu = vecs.mean(axis=0)
    mu /= np.linalg.norm(mu)
    d_data = np.arccos(np.clip(vecs @ mu, -1, 1))
    d_test = np.arccos(np.clip(np.dot(test_vec, mu), -1, 1))
    # +1/+1 keeps p away from exactly 0 (can't claim p < 1/(N+1) from N points)
    return (np.sum(d_data >= d_test) + 1) / (len(d_data) + 1)
 
 
# ---------------------------------------------------------------- method 3
def hpd_pvalue(vecs, test_vec, nside=128, smooth_deg=None):
    """Highest-posterior-density p-value via a smoothed healpix map
    (same idea as 'searched credible level' in GW skymap papers).
 
    Handles elongated / asymmetric / multimodal clouds: p = 1 - (probability
    mass in pixels denser than the test point's pixel).
    """
    if smooth_deg is None:
        # crude bandwidth rule of thumb from the angular spread of the data
        mu = vecs.mean(axis=0); mu /= np.linalg.norm(mu)
        spread = np.degrees(np.std(np.arccos(np.clip(vecs @ mu, -1, 1))))
        smooth_deg = max(spread * len(vecs) ** (-1 / 6), 3 * hp.nside2resol(nside, arcmin=True) / 60)
    m = np.zeros(hp.nside2npix(nside))
    np.add.at(m, hp.vec2pix(nside, *vecs.T), 1.0)
    m = hp.smoothing(m, sigma=np.radians(smooth_deg))
    m = np.clip(m, 0, None)
    m /= m.sum()
    dens_test = m[hp.vec2pix(nside, *test_vec)]
    credible_level = m[m > dens_test].sum()   # e.g. 0.997 -> "outside 3-sigma region"
    return 1.0 - credible_level
# ----------------------------------------------------------------------------

def fig_dipole_dir_results(args, exp):
    ecal = "mid"

    plt.figure(figsize=(16, 10.5))

    for x in range(3):
        for y in range(3):
            dens = [-2, -4, -5][x]
            model = [-2, -2, 0][y]
            gmf = [0, 6, 0][y] # twistX
            sp = y*3+x+1

            res = get_results_dipole(args, exp, 1, model, dens, gmf, ecal)
            resn = res / np.linalg.norm(res, axis=1).reshape(10000, 1)

            nside=32
            b = resn.mean(axis=0)
            b /= np.linalg.norm(b)

            r = np.percentile(np.arccos(np.clip(resn @ b, -1, 1)), 90)
            e1 = np.cross(b, [0, 0, 1.]); e1 /= np.linalg.norm(e1); e2 = np.cross(b, e1)
            t = np.linspace(0, 2*np.pi, 400)
            circ = np.cos(r)*b + np.sin(r)*(np.cos(t)[:, None]*e1 + np.sin(t)[:, None]*e2)

            m = np.bincount(hp.vec2pix(nside, *resn.T), minlength=hp.nside2npix(nside)).astype(float)
            m /= len(resn) *hp.nside2pixarea(nside)

            ttl = r"$s_0=10^{"+str(dens)+r"} \text{Mpc}^{-3}$" if y == 0 else ""
            hp.mollview(m, cmap='Reds', min=0, badcolor='white', bgcolor='white',
                        sub=(3,3,sp), title=ttl)
            ax = plt.gca()

            ax.graticule(dpar=30, dmer=30, color='0.7', lw=0.5)
            ax.projplot(*hp.vec2ang(circ), color='red', lw=1.8)
            ax.projscatter(*hp.vec2ang(b), marker='s', s=90, c='red', edgecolor='k', lw=0.8, zorder=5)

            if exp == "auger":
                dt = np.array([-0.06125621, -0.07736099, -0.0123766])
                datda = hp.vec2ang(dt)
                ax.projscatter(*datda, s=100, color='black')

                print(empirical_pvalue(resn, dt/np.linalg.norm(dt)))
                print(hpd_pvalue(resn, dt/np.linalg.norm(dt), smooth_deg=None))

    # plt.tight_layout()
    plt.show()

def fig_dipole_direction_p(args):
        ecal = "mid"
        bias = 1

        plt.figure(figsize=(5, 5))

        pro = []
        prom = [[],[],[],[],[],[],[],[]]
        nuc = []
        nucm = [[],[],[],[],[],[],[],[]]
    
        dt = np.array([-0.06125621, -0.07736099, -0.0123766])
        for dens in [-2, -3, -4, -5]:
            res = get_results_dipole(args, "auger", bias, 0, dens, -1, ecal)
            resn = res / np.linalg.norm(res, axis=1).reshape(10000, 1)
            pro.append(empirical_pvalue(resn, dt/np.linalg.norm(dt)))

            res = get_results_dipole(args, "auger", bias, -2, dens, -1, ecal)
            resn = res / np.linalg.norm(res, axis=1).reshape(10000, 1)
            nuc.append(empirical_pvalue(resn, dt/np.linalg.norm(dt)))

            for m in range(8):
                res = get_results_dipole(args, "auger", bias, 0, dens, m, ecal)
                resn = res / np.linalg.norm(res, axis=1).reshape(10000, 1)
                prom[m].append(empirical_pvalue(resn, dt/np.linalg.norm(dt)))

                res = get_results_dipole(args, "auger", bias, -2, dens, m, ecal)
                resn = res / np.linalg.norm(res, axis=1).reshape(10000, 1)
                nucm[m].append(empirical_pvalue(resn, dt/np.linalg.norm(dt)))

        dns = [-2, -3, -4, -5]
        plt.scatter(dns, pro, c='C3', lw=2, label='protons, $b_1=1$', marker='+')
        plt.scatter(dns, nuc, c='C0', lw=2, label='nuclei, $b_1=1$', marker='+')

        for m in range(8):
            plt.plot(dns, prom[m], c='C3', lw=2, alpha=1-.1*m)
            plt.plot(dns, nucm[m], c='C0', lw=2, alpha=1-.1*m)

        plt.xlabel(r"$\log(s_0)$")
        plt.ylabel("p-value")

        from matplotlib.ticker import MaxNLocator
        plt.gca().invert_xaxis()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.yscale("log")

        plt.legend()
        plt.tight_layout()
        plt.show()
                        
    


def fig_dipole_results(args, exp):
    gmf = 0
    ecal = "mid"

    plt.figure(figsize=(15, 5))

    for sp in [1, 2, 3]:
        dens = {1:-2, 2:-4, 3:-5}[sp]

        nuc = [np.linalg.norm(v) for v in get_results_dipole(args, exp, 1, -2, dens, gmf, ecal)]
        pro = [np.linalg.norm(v) for v in get_results_dipole(args, exp, 1, 0, dens, gmf, ecal)]
        nuchigh = [np.linalg.norm(v) for v in get_results_dipole(args, exp, 1.7, -2, dens, gmf, ecal)]
        nuclow = [np.linalg.norm(v) for v in get_results_dipole(args, exp, 0, -2, dens, gmf, ecal)]
        prohigh = [np.linalg.norm(v) for v in get_results_dipole(args, exp, 1.7, 0, dens, gmf, ecal)]
        prolow = [np.linalg.norm(v) for v in get_results_dipole(args, exp, 0, 0, dens, gmf, ecal)]

        bns = np.linspace(min(pro), min(max(nuchigh), 2))

        plt.subplot(130+sp)
        plt.title(r"$s_0=10^{"+str(dens)+r"} \text{Mpc}^{-3}$")

        plt.hist(nuc, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
        plt.hist(pro, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
        plt.hist(nuchigh, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
        plt.hist(prohigh, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')
        plt.hist(nuclow, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, linestyle=":", label='nuclei, $b_1=0$')
        plt.hist(prolow, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, linestyle=":", label='protons, $b_1=0$')

        if exp == "auger":
            dt = np.array([-0.06125621, -0.07736099, -0.0123766])
            value = np.linalg.norm(dt)
            plt.vlines(value, 0, 10, color='black', linestyle='--')
            print(np.sum(pro > value) / len(prohigh))
            print(np.sum(nuc > value) / len(prohigh))


        if dens == -2:
            plt.ylabel("p.d.f")
        plt.xlabel(r"$\alpha$")
        if dens == -4:
            plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()

def fig_dipole_mag(args, exp):

    ecal = "mid"

    plt.figure(figsize=(15, 5))

    for sp in [1, 2, 3]:
        dens = {1:-2, 2:-4, 3:-5}[sp]

        nuc = np.array([np.linalg.norm(v) for v in get_results_dipole(args, exp, 1, -2, dens, -1, ecal)])
        pro = np.array([np.linalg.norm(v) for v in get_results_dipole(args, exp, 1, 0, dens, -1, ecal)])

        bns = np.linspace(min(min(pro), min(nuc)), min(max(nuc), 1.5))

        print(f"HHs{dens}")

        # plt.subplot(130+sp)
        plt.subplot(130+sp)
        plt.title(r"$s_0=10^{"+str(dens)+r"} \text{Mpc}^{-3}$")


        plt.hist(nuc, density=True, alpha=.25, bins=bns, color='C0', label='nuclei')
        plt.hist(pro, density=True, alpha=.25, bins=bns, color='C3', label='protons')


        for m in range(8):
            nucm = np.array([np.linalg.norm(v) for v in get_results_dipole(args, exp, 1, -2, dens, m, ecal)])
            prom = np.array([np.linalg.norm(v) for v in get_results_dipole(args, exp, 1, 0, dens, m, ecal)])

            print(np.sum(nucm > np.linalg.norm(np.array([-0.06125621, -0.07736099, -0.0123766]))) / len(nucm))
    
            plt.hist(nucm, density=True, bins=bns, color='C0', alpha=1-.1*m, histtype='step', linewidth=2)
            plt.hist(prom, density=True, bins=bns,  color='C3', alpha=1-.1*m, histtype='step', linewidth=2)


        if exp == "auger":
            dt = np.array([-0.06125621, -0.07736099, -0.0123766])
            value = np.linalg.norm(dt)
            plt.vlines(value, 0, 10, color='black', linestyle='--')
            print(np.sum(pro > value) / len(pro))
            print(np.sum(nuc > value) / len(nuc))

        if dens == -2:
            plt.ylabel("p.d.f")
        plt.xlabel(r"$\alpha$")
        if dens == -4:
            plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", "-data", help="path with interaction data", default="../CRPropa3/build/data")
    parser.add_argument("--croutput-directory", "-cr", help="path with files", default="/mnt/x/uhecr/cr_output")
    return parser.parse_args()

def main():
    args = parse_args()
    global DP 
    DP = mfp.DataParser(args.data_directory)

    fig_mfp_comparisons()               # Fig. 1
    fig_effective_distance()            # Fig. 2
    fig_contours()                      # Fig. 3
    fig_rigidity()                      # Fig. 4
    fig_distances()                     # Fig. 5

                                        # Fig. 6, 7 are in spectrumplot.py
    # fig_general_coefficient()
    fig_largescale()                    # Fig. 8, 9
    # fig_gmfbins()

    fig_large_results(args, "auger")    # Fig. 10
    fig_large_mag(args, "auger")        # Fig. 11, 12
    # fig_large_cal(args, "ideal")

    fig_small_results(args, "auger")    # Fig. 13, 16
    fig_small_mag(args, "auger")        # Fig. 14, 17
    fig_small_cal(args, "auger")        # Fig. 15

    fig_energy_results(args, "auger")   # Fig. 18
    fig_energy_mag(args, "ideal")       # Fig. 19, 20
    # fig_energy_cal(args, "auger")

    fig_dipole_results(args, "auger")   # Fig. 21, 22
    fig_dipole_mag(args, "auger")       # Fig. 23
    fig_dipole_dir_results(args, "auger") # Fig. 24
    fig_dipole_direction_p(args)        # Fig. 25

    fig_large_cal2(args, "auger")       # Fig. 26

if __name__ == "__main__":
    main()
