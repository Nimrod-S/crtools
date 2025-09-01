import argparse

import numpy as np
import matplotlib.pyplot as plt

import propagation
from cosmology import *

IRB_NAME="Gilmore12"

class DataParser:

    INTERESTING_ENERGY_RANGE_eV = (1e18, 1e21)

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
        plt.xlabel("$\gamma$")
        plt.ylabel("$\lambda$ (Mpc)")
        plt.title(f"A={a}")
        # plt.loglog(gammas, mfp_Mpc, label="real", color='royalblue')
        plt.loglog(gammas, e_mfp_Mpc, label="real (effective)", color='mediumseagreen')
        plt.loglog(gammas, model_mfp_Mpc, label="model", color='gray', linestyle='--')
        # plt.loglog(gammas, e_mfp_Mpc, label=f"A={a}")
        # plt.loglog(gammas, model_mfp_Mpc, linestyle='--', color='gray')

        # plt.show()
        # ax = plt.axes()
        # ax.set_xscale("log")
        # plt.plot(gammas, e_mfp_Mpc / mfp_Mpc)
        # plt.show()

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", "-data", help="path with interaction data", default="./CRPropa3/build/data")
    return parser.parse_args()

def main():
    args = parse_args()

    DP = DataParser(args.data_directory)

    
    DP.graph_mfp_by_a(56)
    DP.graph_mfp_by_a(28)
    DP.graph_mfp_by_a(16)
    # DP.graph_mfp_by_a(14)
    DP.graph_mfp_by_a(12)
    plt.legend()
    plt.show()

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


if __name__ == "__main__":
    main()