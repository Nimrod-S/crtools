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
        d = propagation.d1(a, gs, e0)
        d = z2dprop(propagation.deff2z(d))
        d2 = propagation.d2(a, gs, e0)
        d2 = z2dprop(propagation.deff2z(d2))
        plt.plot(e1s, d2, color=color, alpha=1-.7*i/len(e0s), label=f"{int(e0/1e18)} EeV")
        # plt.plot(e1s, d2, color="darkgreen", alpha=1-.7*i/len(e0s))
        # plt.text(e1s[3], d[3], f"{int(e0/1e18)}", color=color, alpha=1-.6*i/len(e0s), backgroundcolor='white')

def plot_side_contours(a, color):
    threshd = np.load(f"thelines{a}.npy")
    
    e1s = np.geomspace(2e19, 2e20)

    for i, thresh in enumerate(threshd):
        plt.plot(e1s, thresh, color=color, alpha=.3+.7*i/len(threshd))


def fig_mfp_comparisons(args):    
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
    # plot_contours([10 ** 19.3, 10 ** 19.4, 10 ** 19.5, 10 ** 19.6, 10 ** 19.7], 14, "darkblue")
    # plot_side_contours(14, "darkred")
    # plt.legend()
    # plt.show()
    # return

    plt.subplot(221)
    plt.ylabel("distance (Mpc)")
    plot_contours(e0s, 56, "darkblue")
    plot_side_contours(56, "darkred")
    # e = np.logspace(19, 20)
    # ee, dm = propagation.maxd(e, 56)
    # plt.plot(ee * 56 * propagation.MP, dm, color="black")

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
    dndrp = propagation.calc_cosmic_ray_rate_density_bonus(2e19, 1e21, zs)
    plt.subplot(131)
    plt.title(r"$E_o>20~\text{EeV}$")
    plt.xlim(0, 800)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("distance (Mpc)")
    plt.ylabel(r"$\psi$ (rays $\text{yr}^{-1}$ $\text{Mpc}^{-3})$")
    plt.plot(z2dprop(zs), dndr, color='C0')
    plt.plot(z2dprop(zs), dndrp, color='C3')

    # plt.figure()
    dndr = propagation.calc_cosmic_ray_rate_density(4e19, 1e21, zs, -2)
    dndrp = propagation.calc_cosmic_ray_rate_density_bonus(4e19, 1e21, zs)
    plt.subplot(132)
    plt.title(r"$E_o>40~\text{EeV}$")
    plt.xlim(0, 600)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("distance (Mpc)")
    plt.plot(z2dprop(zs), dndr, color='C0')
    plt.plot(z2dprop(zs), dndrp, color='C3')

    dndr = propagation.calc_cosmic_ray_rate_density(6e19, 1e21, zs, -2)
    dndrp = propagation.calc_cosmic_ray_rate_density_bonus(6e19, 1e21, zs)
    plt.subplot(133)
    plt.title(r"$E_o>60~\text{EeV}$")
    plt.xlim(0, 300)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("distance (Mpc)")
    plt.plot(z2dprop(zs), dndr, color='C0', label='nuclei')
    plt.plot(z2dprop(zs), dndrp, color='C3', label='protons')

    # dndr = propagation.calc_cosmic_ray_rate_density(8e19, 1e21, zs, -2)
    # dndrp = propagation.calc_cosmic_ray_rate_density_bonus(8e19, 1e21, zs)
    # plt.subplot(224)
    # plt.xlabel("distance (Mpc)")
    # plt.plot(z2dprop(zs), dndr, color='darkblue')
    # plt.plot(z2dprop(zs), dndrp, color='red')
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
    top = get_results(args, f"mfnuc_k{mask}_{calibration}{magname}", exp, bias, model, sdens)
    bot = get_results(args, f"mfpro_k{mask}_{calibration}{magname}", exp, bias, model, sdens)
    return top - bot

def get_results_small(args, exp, bias, model, sdens, gmf, mask, calibration):
    magname = "" if (gmf == -1) else f"_U{gmf}"
    return get_results(args, f"ens16.7_k{mask}_{calibration}{magname}", exp, bias, model, sdens)

def get_results_energy(args, exp, bias, model, sdens, gmf, mask, calibration):
    magname = "" if (gmf == -1) else f"_U{gmf}"
    return get_results(args, f"ec16.7_k{mask}_{calibration}{magname}", exp, bias, model, sdens)

def get_results_dipole(args, exp, bias, model, sdens, gmf):
    magname = "" if (gmf == -1) else f"_U{gmf}"
    if model == -2:
        nm = "e2e42"
    else:
        nm = "e3e42" # STUPID TYPO BY ME WHEN CREATING THESE
    return get_results(args, f"ds16.7_{nm}{magname}", exp, bias, model, sdens)

def fig_large_mag(args, exp):

    ecal = "mid"
    mask = 10

    plt.figure(figsize=(15, 5))

    for dens in [-2, -3, -4]:
        sp = -1-dens

        nuc = get_results_large_vs(args, exp, 1, -2, dens, -1, mask, ecal)
        pro = get_results_large_vs(args, exp, 1, 0, dens, -1, mask, ecal)

        bns = np.linspace(min(pro), max(nuc))

        plt.subplot(130+sp)
        plt.title(r"$s_0=10^{"+str(dens)+r"} \text{Mpc}^{-3}$")


        plt.hist(nuc, density=True, alpha=.25, bins=bns, color='C0', label='nuclei')
        plt.hist(pro, density=True, alpha=.25, bins=bns, color='C3', label='protons')


        for m in range(8):
            nucm = get_results_large_vs(args, exp, 1, -2, dens, m, mask, ecal)
            prom = get_results_large_vs(args, exp, 1, 0, dens, m, mask, ecal)
            print(np.sum(nucm < 1e-2) / len(nucm))
    
            plt.hist(nucm, density=True, bins=bns, color='C0', alpha=1-.1*m, histtype='step', linewidth=2)
            plt.hist(prom, density=True, bins=bns,  color='C3', alpha=1-.1*m, histtype='step', linewidth=2)


        if exp == "auger":
            if mask == 0:
                value = 7.9533602462936095 - 7.9654454965694015
            elif mask == 10:
                value = 7.746544395046049 - 7.748923278654131
                value = 0.11555984427536525 - 0.11793872788344828
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

    for dens in [-2, -3, -4]:
        sp = -1-dens

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
            if mask == 0:
                value = 7.9533602462936095 - 7.9654454965694015
            elif mask == 10:
                value = 7.746544395046049 - 7.748923278654131
                value = 0.11555984427536525 - 0.11793872788344828
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
    gmf = 5

    plt.figure(figsize=(15, 5))

    for dens in [-2, -3, -4]:
        sp = -1-dens

        nuc = get_results_large_vs(args, exp, 1, -2, dens, gmf, mask, ecal)
        pro = get_results_large_vs(args, exp, 1, 0, dens, gmf, mask, ecal)
        nuchigh = get_results_large_vs(args, exp, 1.7, -2, dens, gmf, mask, ecal)
        nuclow = get_results_large_vs(args, exp, 0, -2, dens, gmf, mask, ecal)
        prohigh = get_results_large_vs(args, exp, 1.7, 0, dens, gmf, mask, ecal)
        prolow = get_results_large_vs(args, exp, 0, 0, dens, gmf, mask, ecal)

        bns = np.linspace(min(pro), max(nuchigh))

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
                value = 7.9533602462936095 - 7.9654454965694015
            elif mask == 10:
                value = 7.746544395046049 - 7.748923278654131
                value = 0.11555984427536525 - 0.11793872788344828
            plt.vlines(value, 0, 50, color='black', linestyle='--')

        plt.ylabel("p.d.f")
        plt.xlabel("$T$")
        plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()


def fig_small_mag(args, exp):

    ecal = "high"
    mask = 10

    plt.figure(figsize=(15, 5))

    for dens in [-2, -3, -4]:
        sp = -1-dens

        nuc = get_results_small(args, exp, 1, -2, dens, -1, mask, ecal)
        pro = get_results_small(args, exp, 1, 0, dens, -1, mask, ecal)

        bns = np.linspace(min(nuc), max(pro))

        print(f"HHs{dens}")
        print(np.sum(nuc > 8.977397198232872) / len(nuc))

        plt.subplot(130+sp)
        plt.title(r"$s_0=10^{"+str(dens)+r"} \text{Mpc}^{-3}$")


        plt.hist(nuc, density=True, alpha=.25, bins=bns, color='C0', label='nuclei')
        plt.hist(pro, density=True, alpha=.25, bins=bns, color='C3', label='protons')


        for m in range(8):
            nucm = get_results_small(args, exp, 1, -2, dens, m, mask, ecal)
            prom = get_results_small(args, exp, 1, 0, dens, m, mask, ecal)

            print(np.sum(nucm > 37.98) / len(nucm))
    
            plt.hist(nucm, density=True, bins=bns, color='C0', alpha=1-.1*m, histtype='step', linewidth=2)
            plt.hist(prom, density=True, bins=bns,  color='C3', alpha=1-.1*m, histtype='step', linewidth=2)


        if exp == "auger":
            if mask == 0:
                value = 9.162295334841339
            elif mask == 10:
                value = 8.977397198232872
            plt.vlines(value, 0, 50, color='black', linestyle='--')

        if dens == -2:
            plt.ylabel("p.d.f")
        plt.xlabel("$S_E$")
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

    for dens in [-2, -3, -4]:
        sp = -1-dens

        nuc = get_results_small(args, exp, 1, -2, dens, gmf, mask, "mid")
        pro = get_results_small(args, exp, 1, 0, dens, gmf, mask, "mid")
        nuchigh = get_results_small(args, exp, 1, -2, dens, gmf, mask, "high")
        nuclow = get_results_small(args, exp, 1, -2, dens, gmf, mask, "low")
        prohigh = get_results_small(args, exp, 1, 0, dens, gmf, mask, "high")
        prolow = get_results_small(args, exp, 1, 0, dens, gmf, mask, "low")

        bns = np.linspace(min(nuchigh), max(prolow))

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
                value = 9.162295334841339
            elif mask == 10:
                value = 8.977397198232872
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

    for dens in [-2, -3, -4]:
        sp = -1-dens

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
                value = 9.162295334841339
            elif mask == 10:
                value = 8.977397198232872
            plt.vlines(value, 0, 50, color='black', linestyle='--')

        if dens == -2:
            plt.ylabel("p.d.f")
        plt.xlabel("$S_E$")
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

    for dens in [-2, -3, -4]:
        sp = -1-dens

        nuc = get_results_energy(args, exp, 1, -2, dens, -1, mask, ecal)
        pro = get_results_energy(args, exp, 1, 0, dens, -1, mask, ecal)

        bns = np.linspace(min(pro), min([max(nuc), 3e-4]))


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
                value = 0.00011774532661636003
            elif mask == 10:
                value = 0.00014063331503275217
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

    for dens in [-2, -3, -4]:
        sp = -1-dens

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
                value = 0.00011774532661636003
            elif mask == 10:
                value = 0.00014063331503275217
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

    for dens in [-2, -3, -4]:
        sp = -1-dens

        nuc = get_results_energy(args, exp, 1, -2, dens, gmf, mask, ecal)
        pro = get_results_energy(args, exp, 1, 0, dens, gmf, mask, ecal)
        nuchigh = get_results_energy(args, exp, 1.7, -2, dens, gmf, mask, ecal)
        nuclow = get_results_energy(args, exp, 0, -2, dens, gmf, mask, ecal)
        prohigh = get_results_energy(args, exp, 1.7, 0, dens, gmf, mask, ecal)
        prolow = get_results_energy(args, exp, 0, 0, dens, gmf, mask, ecal)

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
            if mask == 0:
                value = 0.00011774532661636003
            elif mask == 10:
                value = 0.00014063331503275217
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


def dm(vs, which):
    return np.linalg.norm(vs[which])

def vang(vs):
    v1, v2 = vs
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    th = np.acos(np.dot(v1, v2))
    return th * 180 / np.pi

def fig_swing_results(args):
    gmf = 7

    nuc2 = [vang(v) for v in get_results_dipole(args, "2022", 1, -2, -2, gmf)]
    pro2 = [vang(v) for v in get_results_dipole(args, "2022", 1, 0, -2, gmf)]
    nuc4 = [vang(v) for v in get_results_dipole(args, "2022", 1, -2, -4, gmf)]
    pro4 = [vang(v) for v in get_results_dipole(args, "2022", 1, 0, -4, gmf)]

    nuchigh = [vang(v) for v in get_results_dipole(args, "2022", 1.7, -2, -2, gmf)]
    prohigh = [vang(v) for v in get_results_dipole(args, "2022", 1.7, 0, -2, gmf)]
    nuc4high = [vang(v) for v in get_results_dipole(args, "2022", 1.7, -2, -4, gmf)]
    pro4high = [vang(v) for v in get_results_dipole(args, "2022", 1.7, 0, -4, gmf)]
    # WARNING COOL
    nuchigh = [vang(v) for v in get_results_dipole(args, "2022", 1, -2, -2, gmf)]
    prohigh = [vang(v) for v in get_results_dipole(args, "2022", 1, 0, -2, gmf)]
    
    nuclow = [vang(v) for v in get_results_dipole(args, "2022", 0, -2, -2, gmf)]
    
    bns = np.linspace(0, 110)


    plt.subplot(121)
    plt.title(r"$s_0=10^{-2} \text{Mpc}^{-3}$")

    plt.hist(nuc2, density=True, alpha=.25, bins=bns, color="C0", label='nuclei, $b_1=1$')
    plt.hist(pro2, density=True, alpha=.25, bins=bns, color="C3", label='protons, $b_1=1$')
    plt.hist(nuchigh, density=True, alpha=.6, bins=bns, color="C0", histtype="step", linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(prohigh, density=True, alpha=.6, bins=bns, color="C3", histtype="step", linewidth=2, label='protons, $b_1=1.7$')

    value = 91.1762215
    plt.vlines(value, 0, 5e-2, color='black', linestyle='--')
    plt.ylabel("p.d.f")
    plt.xlabel(r"$\Delta\alpha_{2,4}$ [deg]")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # plt.legend()

    plt.subplot(122)
    plt.title(r"$s_0=10^{-4} \text{Mpc}^{-3}$")
        
    bns = np.linspace(0, 110)
    plt.hist(nuc4, density=True, alpha=.25, bins=bns, color="C0", label='nuclei, $b_1=1$')
    plt.hist(pro4, density=True, alpha=.25, bins=bns, color="C3", label='protons, $b_1=1$')
    plt.hist(nuc4high, density=True, alpha=.6, bins=bns, color="C0", histtype="step", linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(pro4high, density=True, alpha=.6, bins=bns, color="C3", histtype="step", linewidth=2, label='protons, $b_1=1.7$')

    value = 91.1762215
    plt.vlines(value, 0, 5e-2, color='black', linestyle='--')
    plt.ylabel("p.d.f")
    plt.xlabel(r"$\Delta\alpha_{2,4}$ [deg]")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend()

    plt.show()


def fig_dipole_results(args, exp):
    mask = 10
    gmf = 0

    plt.figure(figsize=(15, 5))

    for dens in [-2, -3, -4]:
        sp = -1-dens

        nuc = [dm(v, 0) for v in get_results_dipole(args, exp, 1, -2, dens, gmf)]
        pro = [dm(v, 0) for v in get_results_dipole(args, exp, 1, 0, dens, gmf)]
        nuchigh = [dm(v, 0) for v in get_results_dipole(args, exp, 1.7, -2, dens, gmf)]
        nuclow = [dm(v, 0) for v in get_results_dipole(args, exp, 0, -2, dens, gmf)]
        prohigh = [dm(v, 0) for v in get_results_dipole(args, exp, 1.7, 0, dens, gmf)]
        prolow = [dm(v, 0) for v in get_results_dipole(args, exp, 0, 0, dens, gmf)]

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
            if mask == 0:
                value = 9.162295334841339
            elif mask == 10:
                value = 8.977397198232872
            plt.vlines(value, 0, 50, color='black', linestyle='--')

        if dens == -2:
            plt.ylabel("p.d.f")
        plt.xlabel("$S_E$")
        if dens == -4:
            plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", "-data", help="path with interaction data", default="../CRPropa3/build/data")
    parser.add_argument("--croutput-directory", "-cr", help="path with files", default="../cr_output")
    # parser.add_argument("--croutput-directory", "-cr", help="path with files", default="/mnt/x/uhecr/cr_output")
    return parser.parse_args()

def main():
    args = parse_args()
    global DP 
    DP = mfp.DataParser(args.data_directory)

    # fig_mfp_comparisons(args)
    # fig_contours()
    # fig_rigidity()
    # fig_distances()

    # fig_general_coefficient()
    # fig_largescale()
    # fig_gmfbins()

    # fig_large_results(args, "ideal")
    # fig_large_mag(args, "ideal")
    # fig_large_cal(args, "ideal")

    # fig_small_results(args, "auger")
    # fig_small_mag(args, "ideal")
    # fig_small_cal(args, "auger")


    # fig_energy_results(args, "auger")
    # fig_energy_mag(args, "auger")
    # fig_energy_cal(args, "auger")

    fig_swing_results(args)
    # fig_dipole_results(args, "2022")

if __name__ == "__main__":
    main()
