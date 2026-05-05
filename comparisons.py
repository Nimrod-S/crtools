import argparse

import numpy as np
import scipy as sp
import healpy as hp
import matplotlib.pyplot as plt

import propagation
import analysis
import mfp
from cosmology import *

import crpropa


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
    m = np.ones(hp.nside2npix(32)) * hp.UNSEEN
    mft = analysis.BigMatchedFilterTest(m)
    
    v = 1
    for r in mft._regions:
        for ipix in r:
            m[ipix] = v
        v = -v
    
    hp.mollview(m, cbar=False, title="Division of the sky to regions (galactic coordinates)", cmap="magma")
    plt.show()

    return


def get_results_large_vs(args, t, b, exp, bias, model, sdens):
    top = get_results(args, "mf"+t, exp, bias, model, sdens)
    bot = get_results(args, "mf"+b, exp, bias, model, sdens)
    return top - bot

def get_results_small_vs(args, t, b, exp, bias, model, sdens):
    top = get_results(args, "svD1_mid_"+t, exp, bias, model, sdens)
    bot = get_results(args, "svD1_mid_"+b, exp, bias, model, sdens)
    return top - bot

def fig_large_results(args):
    # auger10 chem
    nuc2 = get_results_large_vs(args, "nuc", "pro", "auger10", 1, -2, -2)
    pro2 = get_results_large_vs(args, "nuc", "pro", "auger10", 1, 0, -2)
    nuchigh = get_results_large_vs(args, "nuc", "pro", "auger10", 1.7, -2, -2)
    nuclow = get_results_large_vs(args, "nuc", "pro", "auger10", 0, -2, -2)
    prohigh = get_results_large_vs(args, "nuc", "pro", "auger10", 1.7, 0, -2)
    prolow = get_results_large_vs(args, "nuc", "pro", "auger10", 0, 0, -2)
    nuc3 = get_results_large_vs(args, "nuc", "pro", "auger10", 1, -2, -3)
    pro3 = get_results_large_vs(args, "nuc", "pro", "auger10", 1, 0, -3)
    nuc3high = get_results_large_vs(args, "nuc", "pro", "auger10", 1.7, -2, -3)
    pro3high = get_results_large_vs(args, "nuc", "pro", "auger10", 1.7, 0, -3)
    nuc4 = get_results_large_vs(args, "nuc", "pro", "auger10", 1, -2, -4)
    pro4 = get_results_large_vs(args, "nuc", "pro", "auger10", 1, 0, -4)
    nuc4high = get_results_large_vs(args, "nuc", "pro", "auger10", 1.7, -2, -4)
    pro4high = get_results_large_vs(args, "nuc", "pro", "auger10", 1.7, 0, -4)

    value = -.012842515543194988
    # value = -0.019420056194455027
    print(np.std(nuc2), np.mean(nuc2))

    bns = np.linspace(min(pro2), max(nuchigh))

    plt.subplot(131)
    plt.title(r"$s_0=10^{-2} \text{Mpc}^{-3}$")

    plt.hist(nuc2, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
    plt.hist(pro2, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
    plt.hist(nuchigh, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(prohigh, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')
    plt.hist(nuclow, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, linestyle=":", label='nuclei, $b_1=0$')
    plt.hist(prolow, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, linestyle=":", label='protons, $b_1=0$')
    
    plt.ylabel("p.d.f")
    plt.xlabel("$T$")
    plt.vlines(value, 0, 50, color='black', linestyle='--')
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.subplot(132)
    plt.title(r"$s_0=10^{-3} \text{Mpc}^{-3}$")

    plt.hist(nuc3, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
    plt.hist(pro3, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
    plt.hist(nuc3high, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(pro3high, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')

    plt.ylabel("p.d.f")
    plt.xlabel("$T$")
    plt.vlines(value, 0, 40, color='black', linestyle='--')
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.subplot(133)
    plt.title(r"$s_0=10^{-4} \text{Mpc}^{-3}$")

    plt.hist(nuc4, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
    plt.hist(pro4, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
    plt.hist(nuc4high, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(pro4high, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')

    plt.ylabel("p.d.f")
    plt.xlabel("$T$")
    plt.vlines(value, 0, 40, color='black', linestyle='--')
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # 2022 chem
    nuc2 = get_results_large_vs(args, "nuc", "pro", "2022", 1, -2, -2)
    pro2 = get_results_large_vs(args, "nuc", "pro", "2022", 1, 0, -2)
    nuchigh = get_results_large_vs(args, "nuc", "pro", "2022", 1.7, -2, -2)
    prohigh = get_results_large_vs(args, "nuc", "pro", "2022", 1.7, 0, -2)
    nuclow = get_results_large_vs(args, "nuc", "pro", "2022", 0, -2, -2)
    prolow = get_results_large_vs(args, "nuc", "pro", "2022", 0, 0, -2)
    nuc3 = get_results_large_vs(args, "nuc", "pro", "2022", 1, -2, -3)
    pro3 = get_results_large_vs(args, "nuc", "pro", "2022", 1, 0, -3)
    nuc3high = get_results_large_vs(args, "nuc", "pro", "2022", 1.7, -2, -3)
    pro3high = get_results_large_vs(args, "nuc", "pro", "2022", 1.7, 0, -3)
    nuc4 = get_results_large_vs(args, "nuc", "pro", "2022", 1, -2, -4)
    pro4 = get_results_large_vs(args, "nuc", "pro", "2022", 1, 0, -4)
    nuc4high = get_results_large_vs(args, "nuc", "pro", "2022", 1.7, -2, -4)
    pro4high = get_results_large_vs(args, "nuc", "pro", "2022", 1.7, 0, -4)

    bns = np.linspace(min(prolow), max(nuchigh))

    plt.figure()
    plt.subplot(131)
    plt.title(r"$s_0=10^{-2} \text{Mpc}^{-3}$")

    plt.hist(nuc2, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
    plt.hist(pro2, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
    plt.hist(nuchigh, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(prohigh, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')
    plt.hist(nuclow, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, linestyle=":", label='nuclei, $b_1=0$')
    plt.hist(prolow, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, linestyle=":", label='protons, $b_1=0$')
    
    plt.ylabel("p.d.f")
    plt.xlabel("$T$")
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
    plt.subplot(132)
    plt.title(r"$s_0=10^{-3} \text{Mpc}^{-3}$")

    bns = np.linspace(min(pro3), max(nuc3high))

    plt.hist(nuc3, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
    plt.hist(pro3, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
    plt.hist(nuc3high, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(pro3high, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')
    
    plt.ylabel("p.d.f")
    plt.xlabel("$T$")
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # 2022
    plt.subplot(133)
    plt.title(r"$s_0=10^{-4} \text{Mpc}^{-3}$")

    bns = np.linspace(min(pro4), max(nuc4high))

    plt.hist(nuc4, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
    plt.hist(pro4, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
    plt.hist(nuc4high, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(pro4high, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')

    plt.ylabel("p.d.f")
    plt.xlabel("$T$")
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


    plt.show()

def fig_small_mag(args):
    nuc = get_results_small_vs(args, "nuc", "pro", "2022", 1, -2, -2)
    pro = get_results_small_vs(args, "nuc", "pro", "2022", 1, 0, -2)

    bns = np.linspace(min(nuc), max(pro))

    # value = 76.9648166415027 - 6.367164550279995 # old def
    # value = 115.17208710215527 - 109.99799606832312 # 20ev def
    # value = 116.39946417065039 - 81.18456190681536 # 32ev def

    value = np.float64(27683.819467027555) - np.float64(26556.291798280654) # 32ev

    value = 48.38093733900314 - 94.98215176451049 # with 20 mask
    value = np.float64(116.39946417065039) - np.float64(81.18456190681536) # with 0 mask
    # value = np.float64(94.01456236287595) - np.float64(80.57173047770266) # with neutral mask

    print(np.sum(nuc > value)/10000)
    print(np.sum(pro < value)/10000)
    
    plt.title(r"$s_0=10^{-2} \text{Mpc}^{-3}$")

    plt.hist(nuc, density=True, alpha=.25, bins=bns, color='C0', label='nuclei')
    plt.hist(pro, density=True, alpha=.25, bins=bns, color='C3', label='protons')

    for m in range(8):
        nucm = get_results_small_vs(args, f"nuc_U{m}", f"pro_U{m}", "2022", 1, -2, -2)
        prom = get_results_small_vs(args, f"nuc_U{m}", f"pro_U{m}", "2022", 1, 0, -2)
 
        plt.hist(nucm, density=True, bins=bns, color='C0', alpha=1-.1*m, label=f'nuclei, m={m}', histtype='step', linewidth=2)
        plt.hist(prom, density=True, bins=bns,  color='C3', alpha=1-.1*m, label=f'protons, m={m}', histtype='step', linewidth=2)

        print(f"MODEL {m}")
        print(np.sum(nucm > value)/10000)
        print(np.sum(prom < value)/10000)

    plt.ylabel("p.d.f")
    plt.xlabel("$T$")
    plt.vlines(value, 0, 1e-2, color='black', linestyle='--')
    # plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.show()

def fig_small_results(args):
    # auger chem
    nuc2 = get_results_small_vs(args, "nuc", "pro", "auger", 1, -2, -2)
    pro2 = get_results_small_vs(args, "nuc", "pro", "auger", 1, 0, -2)
    nuchigh = get_results_small_vs(args, "nuc", "pro", "auger", 1.7, -2, -2)
    nuclow = get_results_small_vs(args, "nuc", "pro", "auger", 0, -2, -2)
    prohigh = get_results_small_vs(args, "nuc", "pro", "auger", 1.7, 0, -2)
    prolow = get_results_small_vs(args, "nuc", "pro", "auger", 0, 0, -2)
    nuc3 = get_results_small_vs(args, "nuc", "pro", "auger", 1, -2, -3)
    pro3 = get_results_small_vs(args, "nuc", "pro", "auger", 1, 0, -3)
    nuc3high = get_results_small_vs(args, "nuc", "pro", "auger", 1.7, -2, -3)
    pro3high = get_results_small_vs(args, "nuc", "pro", "auger", 1.7, 0, -3)
    nuc4 = get_results_small_vs(args, "nuc", "pro", "auger", 1, -2, -4)
    pro4 = get_results_small_vs(args, "nuc", "pro", "auger", 1, 0, -4)
    nuc4high = get_results_small_vs(args, "nuc", "pro", "auger", 1.7, -2, -4)
    pro4high = get_results_small_vs(args, "nuc", "pro", "auger", 1.7, 0, -4)

    # nuclow = get_results_small_vs(args, "nuc_U", "pro_U", "auger", 1, -2, -2)
    # prolow = get_results_small_vs(args, "nuc_U", "pro_U", "auger10", 1, 0, -2)

    value = 76.9648166415027 - 6.367164550279995
    # value = 115.17208710215527 - 109.99799606832312
    # value = 116.39946417065039 - 81.18456190681536 # 32ev dev

    value = np.float64(27683.819467027555) - np.float64(26556.291798280654) # 32ev

    value = 48.38093733900314 - 94.98215176451049 # new def

    print(np.std(nuc2), np.mean(nuc2))

    # print(min(nuc2), max(nuc2))
    # print(min(pro2), max(pro2))
    # print(min(nuchigh), max(nuchigh))
    # print(min(nuclow), max(nuclow))
    # print(min(prohigh), max(prohigh))
    # print(min(prolow), max(prolow))
    # print(min(nuc3), max(nuc3))
    # print(min(pro3), max(pro3))
    # print(min(nuc3high), max(nuc3high))
    # print(min(pro3high), max(pro3high))
    # print(min(nuc4), max(nuc4))
    # print(min(pro4), max(pro4))
    # print(min(nuc4high), max(nuc4high))
    # print(min(pro4high), max(pro4high))

    bns = np.linspace(min(nuchigh), max(prolow))

    plt.subplot(131)
    plt.title(r"$s_0=10^{-2} \text{Mpc}^{-3}$")

    plt.hist(nuc2, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
    plt.hist(pro2, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
    plt.hist(nuchigh, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(prohigh, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')
    plt.hist(nuclow, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, linestyle=":", label='nuclei, $b_1=0$')
    plt.hist(prolow, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, linestyle=":", label='protons, $b_1=0$')
    
    plt.ylabel("p.d.f")
    plt.xlabel("$T$")
    plt.vlines(value, 0, 4e-3, color='black', linestyle='--')
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.subplot(132)
    plt.title(r"$s_0=10^{-3} \text{Mpc}^{-3}$")

    plt.hist(nuc3, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
    plt.hist(pro3, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
    plt.hist(nuc3high, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(pro3high, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')

    plt.ylabel("p.d.f")
    plt.xlabel("$T$")
    plt.vlines(value, 0, 4e-3, color='black', linestyle='--')
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.subplot(133)
    plt.title(r"$s_0=10^{-4} \text{Mpc}^{-3}$")

    plt.hist(nuc4, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
    plt.hist(pro4, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
    plt.hist(nuc4high, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(pro4high, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')

    plt.ylabel("p.d.f")
    plt.xlabel("$T$")
    plt.vlines(value, 0, 4e-3, color='black', linestyle='--')
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.show()
    return
    # 2022 chem
    nuc2 = get_results_small_vs(args, "nuc", "pro", "2022", 1, -2, -2)
    pro2 = get_results_small_vs(args, "nuc", "pro", "2022", 1, 0, -2)
    nuchigh = get_results_small_vs(args, "nuc", "pro", "2022", 1.7, -2, -2)
    prohigh = get_results_small_vs(args, "nuc", "pro", "2022", 1.7, 0, -2)
    nuclow = get_results_small_vs(args, "nuc", "pro", "2022", 0, -2, -2)
    prolow = get_results_small_vs(args, "nuc", "pro", "2022", 0, 0, -2)
    nuc3 = get_results_small_vs(args, "nuc", "pro", "2022", 1, -2, -3)
    pro3 = get_results_small_vs(args, "nuc", "pro", "2022", 1, 0, -3)
    nuc3high = get_results_small_vs(args, "nuc", "pro", "2022", 1.7, -2, -3)
    pro3high = get_results_small_vs(args, "nuc", "pro", "2022", 1.7, 0, -3)
    nuc4 = get_results_small_vs(args, "nuc", "pro", "2022", 1, -2, -4)
    pro4 = get_results_small_vs(args, "nuc", "pro", "2022", 1, 0, -4)
    nuc4high = get_results_small_vs(args, "nuc", "pro", "2022", 1.7, -2, -4)
    pro4high = get_results_small_vs(args, "nuc", "pro", "2022", 1.7, 0, -4)

    # WOW
    nuclow = get_results_small_vs(args, "nuc_U", "pro_U", "2022", 1, -2, -2)
    prolow = get_results_small_vs(args, "nuc_U", "pro_U", "2022", 1, 0, -2)

    # bns = np.linspace(min(prolow), max(nuchigh))
    bns = np.linspace(min(nuchigh), max(pro4))

    plt.figure()
    plt.subplot(131)
    plt.title(r"$s_0=10^{-2} \text{Mpc}^{-3}$")

    plt.hist(nuc2, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
    plt.hist(pro2, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
    plt.hist(nuchigh, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(prohigh, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')
    plt.hist(nuclow, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, linestyle=":", label='nuclei, $b_1=0$')
    plt.hist(prolow, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, linestyle=":", label='protons, $b_1=0$')
    
    plt.ylabel("p.d.f")
    plt.xlabel("$T$")
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
    plt.subplot(132)
    plt.title(r"$s_0=10^{-3} \text{Mpc}^{-3}$")

    bns = np.linspace(min(pro3), max(nuc3high))
    bns = np.linspace(min(nuc3high), max(pro3))

    plt.hist(nuc3, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
    plt.hist(pro3, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
    plt.hist(nuc3high, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(pro3high, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')
    
    plt.ylabel("p.d.f")
    plt.xlabel("$T$")
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # 2022
    plt.subplot(133)
    plt.title(r"$s_0=10^{-4} \text{Mpc}^{-3}$")

    bns = np.linspace(min(nuc4), max(pro4))

    plt.hist(nuc4, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
    plt.hist(pro4, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
    plt.hist(nuc4high, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(pro4high, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')

    plt.ylabel("p.d.f")
    plt.xlabel("$T$")
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.figure()
    nuclow = get_results_small_vs(args, "Anuc_U", "Apro_U", "2022", 1, -2, -2)
    prolow = get_results_small_vs(args, "Anuc_U", "Apro_U", "2022", 1, 0, -2)
    bns = np.linspace(min(nuclow), max(prolow))
    plt.hist(nuclow, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, linestyle=":", label='nuclei, $b_1=0$')
    plt.hist(prolow, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, linestyle=":", label='protons, $b_1=0$')

    plt.figure()
    nuclow = get_results_small_vs(args, "nuc_U", "pro_U", "auger10", 1, -2, -2)
    prolow = get_results_small_vs(args, "nuc_U", "pro_U", "auger10", 1, 0, -2)
    bns = np.linspace(min(nuclow), max(prolow))
    plt.hist(nuclow, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, linestyle=":", label='nuclei, $b_1=0$')
    plt.hist(prolow, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, linestyle=":", label='protons, $b_1=0$')


    plt.show()

def fig_energy_results(args):
    # Auger10
    nuc2 = get_results(args, "ec16.5_e2e4", "auger10", 1, -2, -2)
    pro2 = get_results(args, "ec16.5_e2e4", "auger10", 1, 0, -2)
    nuchigh = get_results(args, "ec16.5_e2e4", "auger10", 1.7, -2, -2)
    prohigh = get_results(args, "ec16.5_e2e4", "auger10", 1.7, 0, -2)
    value = 0.00013479492048366958
    bns = np.linspace(min(pro2), max(nuchigh))

    plt.subplot(111)  
    plt.title(r"$s_0=10^{-2} \text{Mpc}^{-3}$")

    plt.hist(nuc2, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
    plt.hist(pro2, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
    plt.hist(nuchigh, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(prohigh, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')

    plt.vlines(value, 0, 60000, color='black', linestyle='--')

    plt.ylabel("p.d.f")
    plt.xlabel("$C_{2,4}$")
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.figure()
    plt.subplot(121)
    plt.title(r"$s_0=10^{-2} \text{Mpc}^{-3}$")

    nuc2 = get_results(args, "ec16.5_e2e4", "2022", 1, -2, -2)
    pro2 = get_results(args, "ec16.5_e2e4", "2022", 1, 0, -2)
    nuchigh = get_results(args, "ec16.5_e2e4", "2022", 1.7, -2, -2)
    prohigh = get_results(args, "ec16.5_e2e4", "2022", 1.7, 0, -2)

    bns = np.linspace(min(pro2), max(nuchigh))

    plt.hist(nuc2, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
    plt.hist(pro2, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
    plt.hist(nuchigh, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(prohigh, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')
    
    plt.ylabel("p.d.f")
    plt.xlabel("$C_{2,4}$")
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.subplot(122)
    plt.title(r"$s_0=10^{-4} \text{Mpc}^{-3}$")
    
    nuc4 = get_results(args, "ec16.5_e2e4", "2022", 1, -2, -4)
    pro4 = get_results(args, "ec16.5_e2e4", "2022", 1, 0, -4)
    nuc4high = get_results(args, "ec16.5_e2e4", "2022", 1.7, -2, -4)
    pro4high = get_results(args, "ec16.5_e2e4", "2022", 1.7, 0, -4)
    # bns = np.linspace(min(pro4), max(nuc4high))

    plt.hist(nuc4, density=True, alpha=.25, bins=bns, color='C0', label='nuclei, $b_1=1$')
    plt.hist(pro4, density=True, alpha=.25, bins=bns, color='C3', label='protons, $b_1=1$')
    plt.hist(nuc4high, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(pro4high, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, label='protons, $b_1=1.7$')

    plt.ylabel("p.d.f")
    plt.xlabel("$C_{2,4}$")
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.show()

def vang(vs):
    v1, v2 = vs
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    th = np.acos(np.dot(v1, v2))
    return th * 180 / np.pi

def fig_swing_results(args):
    nuc2 = [vang(v) for v in get_results(args, "ds_e2e4", "2022", 1, -2, -2)]
    pro2 = [vang(v) for v in get_results(args, "ds_e2e4", "2022", 1, 0, -2)]
    nuc4 = [vang(v) for v in get_results(args, "ds_e2e4", "2022", 1, -2, -4)]
    pro4 = [vang(v) for v in get_results(args, "ds_e2e4", "2022", 1, 0, -4)]

    nuchigh = [vang(v) for v in get_results(args, "ds_e2e4", "2022", 1.7, -2, -2)]
    prohigh = [vang(v) for v in get_results(args, "ds_e2e4", "2022", 1.7, 0, -2)]
    nuc4high = [vang(v) for v in get_results(args, "ds_e2e4", "2022", 1.7, -2, -4)]
    pro4high = [vang(v) for v in get_results(args, "ds_e2e4", "2022", 1.7, 0, -4)]
    # WARNING COOL
    nuchigh = [vang(v) for v in get_results(args, "ds_U_e2e4", "2022", 1, -2, -2)]
    prohigh = [vang(v) for v in get_results(args, "ds_U_e2e4", "2022", 1, 0, -2)]
    
    nuclow = [vang(v) for v in get_results(args, "ds_e2e4", "2022", 0, -2, -2)]
    
    bns = np.linspace(0, 60)


    plt.subplot(121)
    plt.title(r"$s_0=10^{-2} \text{Mpc}^{-3}$")

    plt.hist(nuc2, density=True, alpha=.25, bins=bns, color="C0", label='nuclei, $b_1=1$')
    plt.hist(pro2, density=True, alpha=.25, bins=bns, color="C3", label='protons, $b_1=1$')
    plt.hist(nuchigh, density=True, alpha=.6, bins=bns, color="C0", histtype="step", linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(prohigh, density=True, alpha=.6, bins=bns, color="C3", histtype="step", linewidth=2, label='protons, $b_1=1.7$')

    plt.ylabel("p.d.f")
    plt.xlabel(r"$\Delta\alpha_{2,4}$ [deg]")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend()

    plt.subplot(122)
    plt.title(r"$s_0=10^{-4} \text{Mpc}^{-3}$")
        
    bns = np.linspace(0, 100)
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

def fig_lvt_results(args):
    
    nuc2 = get_results(args, "lv16.5", "auger10", 1, -2, -2)
    pro2 = get_results(args, "lv16.5", "auger10", 1, 0, -2)
    nuc3 = get_results(args, "lv16.5", "auger10", 1, -2, -3)
    nuc4 = get_results(args, "lv16.5", "auger10", 1, -2, -4)
    nuc5 = get_results(args, "lv16.5", "auger10", 1, -2, -5)
    pro3 = get_results(args, "lv16.5", "auger10", 1, 0, -3)
    pro4 = get_results(args, "lv16.5", "auger10", 1, 0, -4)
    pro5 = get_results(args, "lv16.5", "auger10", 1, 0, -5)

    nuchigh = get_results(args, "lv16.5", "auger10", 1.7, -2, -2)
    prohigh = get_results(args, "lv16.5", "auger10", 1.7, 0, -2)
    nuc4high = get_results(args, "lv16.5", "auger10", 1.7, -2, -4)
    pro4high = get_results(args, "lv16.5", "auger10", 1.7, 0, -4)
    nuclow = get_results(args, "lv16.5", "auger10", 0, -2, -2)
    prolow = get_results(args, "lv16.5", "auger10", 0, 0, -2)

    value = 1.20018778976404
    pvalue = np.sum(pro5 < value) / len(pro5)
    print(pvalue)
    print(np.mean(nuc2))
    print(np.std(nuc2))

    bns = np.linspace(1.1, 1.6)

    plt.subplot(121)
    plt.title(r"$s_0=10^{-2} \text{Mpc}^{-3}$")
    plt.hist(nuc2, density=True, alpha=.25, bins=bns, color="C0", label='nuclei, $b_1=1$')
    plt.hist(pro2, density=True, alpha=.25, bins=bns, color="C3", label='protons, $b_1=1$')
    plt.hist(nuchigh, density=True, alpha=.6, bins=bns, color="C0", histtype="step", linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(prohigh, density=True, alpha=.6, bins=bns, color="C3", histtype="step", linewidth=2, label='protons, $b_1=1.7$')
    plt.hist(nuclow, density=True, alpha=.6, bins=bns, color="C0", histtype="step", linestyle=":", linewidth=2, label='nuclei, $b_1=0$')
    plt.hist(prolow, density=True, alpha=.6, bins=bns, color="C3", histtype="step", linestyle=":",  linewidth=2, label='protons, $b_1=0$')

    plt.vlines(value, 0, 14, color='black', linestyle='--')
    plt.ylabel("p.d.f")
    plt.xlabel(r"$\xi$")
    plt.legend()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.subplot(122)
    plt.title(r"$s_0=10^{-4} \text{Mpc}^{-3}$")
    plt.hist(nuc4, density=True, alpha=.25, bins=bns, color="C0", label='nuclei, $b_1=1$')
    plt.hist(pro4, density=True, alpha=.25, bins=bns, color="C3", label='protons, $b_1=1$')
    plt.hist(nuc4high, density=True, alpha=.6, bins=bns, color="C0", histtype="step", linewidth=2, label='nuclei, $b_1=1.7$')
    plt.hist(pro4high, density=True, alpha=.6, bins=bns, color="C3", histtype="step", linewidth=2, label='protons, $b_1=1.7$')

    plt.vlines(value, 0, 12, color='black', linestyle='--')
    plt.ylabel("p.d.f")
    plt.xlabel(r"$\xi$")
    plt.legend()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.figure()
    plt.title("protons, $b_1=1$")
    plt.hist(pro2, density=True, alpha=.6, bins=bns, color="C0", histtype="step", linewidth=2, label=r"$s_0=10^{-2} \text{Mpc}^{-3}$")
    plt.hist(pro3, density=True, alpha=.6, bins=bns, color="C2", histtype="step", linewidth=2, label=r"$s_0=10^{-3} \text{Mpc}^{-3}$")
    plt.hist(pro4, density=True, alpha=.6, bins=bns, color="C4", histtype="step", linewidth=2, label=r"$s_0=10^{-4} \text{Mpc}^{-3}$")
    plt.hist(pro5, density=True, alpha=.6, bins=bns, color="C5", histtype="step", linewidth=2, label=r"$s_0=10^{-5} \text{Mpc}^{-3}$")
    
    plt.vlines(value, 0, 14, color='black', linestyle='--')
    plt.ylabel("p.d.f")
    plt.xlabel(r"$\xi$")
    plt.legend()    
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

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

    # fig_mfp_comparisons(args)
    # fig_contours()
    # fig_rigidity()
    # fig_distances()

    # fig_small_results(args)
    fig_small_mag(args)
    # fig_energy_results(args)
    # fig_swing_results(args)
    # fig_lvt_results(args)

if __name__ == "__main__":
    main()
