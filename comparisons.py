import argparse

import numpy as np
import scipy as sp
import healpy as hp
import matplotlib.pyplot as plt

import propagation
import analysis
from cosmology import *

import crpropa


IRB_NAME="Gilmore12"

class DataParser:

    INTERESTING_ENERGY_RANGE_eV = (5e18, 1e21)

    def __init__(self, path):
        self._base_path = path
        self._load_stable_isotopes()
        self._load_photodisintegration_rates()
        self._load_brs()
        self._calculate_effective_rates()

    def _evaluate_prod_mass(self, p):
        #      1: He4
        #     10: He3
        #    100: H3
        #   1000: H2
        #  10000: p
        # 100000: n
        m = 0
        masses = [4, 3, 3, 2, 1, 1]
        for mass in masses:
            m += (p % 10) * mass
            p //= 10
        return m

    def _read_file(self, path):
        full_name = self._base_path + "/" + path
        with open(full_name, "r") as f:
            d = f.read()
        return d
    
    def _load_stable_isotopes(self):
        f = self._read_file("isotopes-stable.txt")
        lines = f.splitlines()[3:] # First three are comments

        # (Z, N, A)
        self._isotopes = [tuple(map(int, l.split())) for l in lines]
        # self._isotopes = [i for i in self._isotopes if i[2] > 9] TODO
    
    def _is_stable(self, z, n):
        return (z, n, z + n) in self._isotopes # In the future, if we just take one for each A, this can be via O(1) lookup
    
    def _get_stable(self, a):
        return [i for i in self._isotopes if i[2] == a]

    def _load_photodisintegration_rates(self):
        cmb = self._read_file("Photodisintegration/rate_CMB.txt")
        cmb_lines = cmb.splitlines()[3:] # First three are comments

        irb = self._read_file("Photodisintegration/rate_IRB_" + IRB_NAME + ".txt")
        irb_lines = irb.splitlines()[3:]

        sol = self._read_file("Photodisintegration/rate_SolarPhotonField.txt")
        sol_lines = sol.splitlines()[3:]
        
        self._pd_rates_gammas = np.logspace(6, 14, 201) # These are the lorentz factor for each rate value in the data
        self._pd_rates_Mpc_cmb = {}
        self._pd_rates_Mpc_irb = {}
        self._pd_rates_Mpc_sol = {}

        for entry in [tuple(l.split()) for l in cmb_lines]:
            z, n = int(entry[0]), int(entry[1])
            if not self._is_stable(z, n):
                continue
            self._pd_rates_Mpc_cmb[(z, n)] = np.fromiter(entry[2:], float)
        for entry in [tuple(l.split()) for l in irb_lines]: # There is definitely a more correct and elegant way of doing this
            z, n = int(entry[0]), int(entry[1])
            if not self._is_stable(z, n):
                continue
            self._pd_rates_Mpc_irb[(z, n)] = np.fromiter(entry[2:], float)
        for entry in [tuple(l.split()) for l in sol_lines]: # There is definitely a more correct and elegant way of doing this
            z, n = int(entry[0]), int(entry[1])
            if not self._is_stable(z, n):
                continue
            self._pd_rates_Mpc_sol[(z, n)] = np.fromiter(entry[2:], float)
    
    def _load_brs(self):
        cmb = self._read_file("Photodisintegration/branching_CMB.txt")
        cmb_lines = cmb.splitlines()[3:]

        irb = self._read_file("Photodisintegration/branching_IRB_" + IRB_NAME + ".txt")
        irb_lines = irb.splitlines()[3:]

        self._pd_brs_cmb = {}
        self._pd_brs_irb = {}

        for entry in [tuple(l.split()) for l in cmb_lines]:
            z, n, prod = int(entry[0]), int(entry[1]), int(entry[2])
            if not self._is_stable(z, n):
                continue
            if (z, n) not in self._pd_brs_cmb.keys():
                self._pd_brs_cmb[(z, n)] = {}
            self._pd_brs_cmb[(z, n)][prod] = np.fromiter(entry[3:], float)

        for entry in [tuple(l.split()) for l in irb_lines]:
            z, n, prod = int(entry[0]), int(entry[1]), int(entry[2])
            if not self._is_stable(z, n):
                continue
            if (z, n) not in self._pd_brs_irb.keys():
                self._pd_brs_irb[(z, n)] = {}
            self._pd_brs_irb[(z, n)][prod] = np.fromiter(entry[3:], float)

    def _calculate_effective_rates(self):
        self._effective_pd_rates_Mpc = {}
        for zn in self._pd_brs_cmb.keys():
            # background_ratio = self._pd_rates_Mpc_cmb[zn] / (self._pd_rates_Mpc_cmb[zn] + self._pd_rates_Mpc_irb[zn])
            effective_rate = np.zeros(201)
            for prod in self._pd_brs_cmb[zn].keys():
                effective_rate += self._evaluate_prod_mass(prod) * self._pd_brs_cmb[zn][prod] * self._pd_rates_Mpc_cmb[zn]
            for prod in self._pd_brs_irb[zn].keys():
                effective_rate += self._evaluate_prod_mass(prod) * self._pd_brs_irb[zn][prod] * self._pd_rates_Mpc_irb[zn]

            self._effective_pd_rates_Mpc[zn] = effective_rate


    def _get_relevant_lorentz_range(self, a):
        min_lorentz = self.INTERESTING_ENERGY_RANGE_eV[0] / (1e9 * a)
        max_lorentz = self.INTERESTING_ENERGY_RANGE_eV[1] / (1e9 * a)

        return np.where((self._pd_rates_gammas > min_lorentz) & (self._pd_rates_gammas < max_lorentz))

    def graph_br_factors(self, g):
        best_g_index = np.searchsorted(self._pd_rates_gammas, g)
        
        datapoints = []
        for zn in self._pd_brs_cmb.keys():
            x = 0
            background_factor = self._pd_rates_Mpc_cmb[zn][best_g_index] / (self._pd_rates_Mpc_cmb[zn][best_g_index] + self._pd_rates_Mpc_irb[zn][best_g_index])
            for prod in self._pd_brs_cmb[zn].keys():
                x += self._evaluate_prod_mass(prod) * self._pd_brs_cmb[zn][prod][best_g_index] * background_factor
                #if prod == 100000 or prod == 10000:
                    #x += self._pd_brs_cmb[zn][prod][best_g_index] * background_factor
            for prod in self._pd_brs_irb[zn].keys():
                x += self._evaluate_prod_mass(prod) * self._pd_brs_irb[zn][prod][best_g_index] * (1 - background_factor)
                #if prod == 100000 or prod == 10000:
                    #x += self._pd_brs_irb[zn][prod][best_g_index] * (1 - background_factor)
            #x = 1 / (2-x)
            datapoints.append((zn[0] + zn[1], 1/x))

        data_a, data_br = zip(*datapoints)


        alist = list(set(data_a))
        alist.sort()
        alist = np.array(alist)

        raise ValueError("Stuff not implemented")
        popt, pcov = curve_fit(power, np.array(data_a), data_br, bounds=([0, 0.5], [np.inf, 0.51]))
        power_fit = power(alist, popt[0], popt[1])
        print(popt)

        ax = plt.axes()
        plt.xlabel("A")
        plt.ylabel("BR 1 nucleon")
        plt.scatter(data_a, data_br)
        plt.plot(alist, power_fit)
        #plt.plot(alist, 0.5 * alist ** 0.2)
        plt.show()
        

    def graph_mfp_by_a(self, a):
        stables = self._get_stable(a)
        if len(stables) == 0:
            print(f"<bad A (no stable isotopes) {a}>")
            return
        elif len(stables) > 1:
            print(f"<bad A (too many stable isotopes) {a}>")
            #return
        zn = stables[0][0], stables[0][1]

        rates_Mpc = self._pd_rates_Mpc_cmb[zn]+self._pd_rates_Mpc_irb[zn]
        e_rates_Mpc = self._effective_pd_rates_Mpc[zn]

        relevant_indices = self._get_relevant_lorentz_range(a)
        gammas = self._pd_rates_gammas[relevant_indices]
        mfp_Mpc = 1 / rates_Mpc[relevant_indices]
        e_mfp_Mpc = 1 / e_rates_Mpc[relevant_indices]

        model_mfp_Mpc = np.array([propagation.mfp1(a, g) for g in gammas])

        # ax = plt.axes()
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        # plt.figure()
        
        plt.title(f"A={a}")
        # plt.loglog(gammas, mfp_Mpc, label="real", color='royalblue')
        # plt.loglog(gammas, e_mfp_Mpc, label="real (effective)", color='mediumseagreen')
        plt.loglog(gammas, e_mfp_Mpc, label="real (effective)", color='darkred')
        plt.loglog(gammas, model_mfp_Mpc, label="model", color='gray', linestyle='--')
        # plt.loglog(gammas, e_mfp_Mpc, label=f"A={a}")
        # plt.loglog(gammas, model_mfp_Mpc, linestyle='--', color='gray')

        # plt.show()
        # ax = plt.axes()
        # ax.set_xscale("log")
        # plt.plot(gammas, e_mfp_Mpc / mfp_Mpc)
        # plt.show()

    def mfp(self, a, g):
        stables = self._get_stable(a)
        zn = stables[0][0], stables[0][1]
        gammas = self._pd_rates_gammas
        e_rates_Mpc = self._effective_pd_rates_Mpc[zn]
        ii = sp.interpolate.interp1d(gammas, e_rates_Mpc)
        return 1 / ii(g)


    def graph_mfp_by_g(self, g, graph=True):
        
        best_g_index = np.searchsorted(self._pd_rates_gammas, g)
        #datapoints = [(zn[0] + zn[1], self._pd_rates_Mpc_cmb[zn][best_g_index] + self._pd_rates_Mpc_irb[zn][best_g_index]) for zn in self._pd_rates_Mpc_cmb.keys()]
        datapoints = [(zn[0] + zn[1], self._effective_pd_rates_Mpc[zn][best_g_index]) for zn in self._effective_pd_rates_Mpc.keys()]
        data_a, data_rate_Mpc = zip(*datapoints)
        data_mfp_Mpc = 1 / np.array(data_rate_Mpc)

        alist = list(set(data_a))
        alist.sort()
        alist = np.array(alist)
        model_mfp_Mpc = propagation.mfp1(alist, g)

        raise ValueError("Stuff not implemented")

        popt, pcov = curve_fit(power, np.array(data_a)[3:], data_mfp_Mpc[3:], bounds=([0, -1.1], [np.inf, -1]))
        power_fit = power(alist, popt[0], popt[1])
        print(popt)
        
        if graph:
            ax = plt.axes()
            ax.set_yscale("log")
            plt.xlabel("A")
            plt.ylabel("rate (1/Mpc)")
            plt.scatter(data_a, data_mfp_Mpc)
            plt.plot(alist, model_mfp_Mpc, color='red')
            plt.plot(alist, power_fit, color='green')
            plt.show()

        return popt[0], popt[1]


def get_results(args, name, exp, bias, model, sdens):
    d = args.croutput_directory + "/results/"

    fname = f"{name}_{exp}_b{bias}_m{model}_s{sdens}.npy"

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
        l = np.array([DP.mfp(a, g) for g in gs])
        d2 = propagation.d2(a, gs, e0, l)
        d2 = z2dprop(propagation.deff2z(d2))
        plt.plot(e1s, d, color=color, alpha=1-.7*i/len(e0s), label=f"{int(e0/1e18)} EeV")
        # plt.plot(e1s, d2, color="darkgreen", alpha=1-.7*i/len(e0s))
        # plt.text(e1s[3], d[3], f"{int(e0/1e18)}", color=color, alpha=1-.6*i/len(e0s), backgroundcolor='white')

def plot_side_contours(a, color):
    threshd = np.load(f"thelines{a}.npy")
    
    e1s = np.geomspace(2e19, 2e20)

    for i, thresh in enumerate(threshd):
        plt.plot(e1s, thresh, color=color, alpha=.3+.7*i/len(threshd))


def fig_mfp_comparisons(args):    
    DP = DataParser(args.data_directory)

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

    # plot_contours([10 ** 19.3, 10 ** 19.4, 10 ** 19.5, 10 ** 19.6, 10 ** 19.7], 14, "darkblue")
    # plot_side_contours(14, "darkred")
    # plt.legend()
    # plt.show()
    # return

    plt.subplot(221)
    plt.ylabel("distance (Mpc)")
    plot_contours([2e19, 4e19, 6e19, 8e19, 1e20], 56, "darkblue")
    plot_side_contours(56, "darkred")

    plt.subplot(222)
    plot_contours([2e19, 4e19, 6e19, 8e19, 1e20], 28, "darkblue")
    plot_side_contours(28, "darkred")

    plt.subplot(223)
    plt.xlabel("$E_s$ (eV)")
    plt.ylabel("distance (Mpc)")
    plot_contours([2e19, 4e19, 6e19, 8e19, 1e20], 16, "darkblue")
    plot_side_contours(16, "darkred")

    plt.subplot(224)
    plt.xlabel("$E_s$ (eV)")
    plot_contours([2e19, 4e19, 6e19, 8e19, 1e20], 12, "darkblue")
    plot_side_contours(12, "darkred")

    plt.legend()
    plt.show()
    return

def fig_rigidity():
    r = np.logspace(17.5, 19.4)
    r = np.linspace(10 ** 17.5, 10 ** 19.4, 3000)
    r2 = propagation.get_r_dist(2e19, r, -2)
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
    i = np.searchsorted(r2cum, 0.01)
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
    # plt.hist(nuclow, density=True, alpha=.6, bins=bns, color='C0', histtype='step', linewidth=2, linestyle=":", label='nuclei, $b_1=0$')
    # plt.hist(prolow, density=True, alpha=.6, bins=bns, color='C3', histtype='step', linewidth=2, linestyle=":", label='protons, $b_1=0$')
    
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
    parser.add_argument("--data-directory", "-data", help="path with interaction data", default="/home/nimrod/physics/uhecr/CRPropa3/build/data")
    parser.add_argument("--croutput-directory", "-cr", help="path with files", default="../cr_output")
    return parser.parse_args()

def main():
    args = parse_args()
    global DP 
    DP = DataParser(args.data_directory)

    # fig_mfp_comparisons(args)
    # fig_contours()
    # fig_rigidity()
    # fig_distances()

    # fig_large_results(args)
    # fig_energy_results(args)
    # fig_swing_results(args)
    # fig_lvt_results(args)

if __name__ == "__main__":
    main()