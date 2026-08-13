import numpy as np
import scipy as sp
import healpy as hp
import matplotlib.pyplot as plt

from cosmology import *

class DataParser:

    INTERESTING_ENERGY_RANGE_eV = (5e18, 1e21)

    def __init__(self, path, irb_name="Gilmore12"):
        self._base_path = path
        self._irb_name = irb_name
        self._load_stable_isotopes()
        self._load_photodisintegration_rates()
        self._load_brs()
        self._calculate_effective_rates()
        self._create_interp_func()

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

        irb = self._read_file("Photodisintegration/rate_IRB_" + self._irb_name + ".txt")
        irb_lines = irb.splitlines()[3:]
        
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
    
    def _load_brs(self):
        cmb = self._read_file("Photodisintegration/branching_CMB.txt")
        cmb_lines = cmb.splitlines()[3:]

        irb = self._read_file("Photodisintegration/branching_IRB_" + self._irb_name + ".txt")
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

    def _create_interp_func(self):
        self._rate_interp_Mpc = {}
        for zn in self._effective_pd_rates_Mpc.keys():
            self._rate_interp_Mpc[zn] = sp.interpolate.interp1d(self._pd_rates_gammas, self._effective_pd_rates_Mpc[zn])

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
        

    def graph_mfp_by_a(self, a, c='darkred'):
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
        
        plt.title(f"A={a}")
        plt.loglog(gammas * a * .938e9, e_mfp_Mpc, label=self._irb_name, color=c)


    def mfp(self, a, g):
        stables = self._get_stable(a)
        zn = stables[0][0], stables[0][1]
        return 1 / self._rate_interp_Mpc[zn](g)


    def graph_mfp_by_g(self, g, graph=True):
        
        best_g_index = np.searchsorted(self._pd_rates_gammas, g)
        #datapoints = [(zn[0] + zn[1], self._pd_rates_Mpc_cmb[zn][best_g_index] + self._pd_rates_Mpc_irb[zn][best_g_index]) for zn in self._pd_rates_Mpc_cmb.keys()]
        datapoints = [(zn[0] + zn[1], self._effective_pd_rates_Mpc[zn][best_g_index]) for zn in self._effective_pd_rates_Mpc.keys()]
        data_a, data_rate_Mpc = zip(*datapoints)
        data_mfp_Mpc = 1 / np.array(data_rate_Mpc)

        alist = list(set(data_a))
        alist.sort()
        alist = np.array(alist)
        # model_mfp_Mpc = propagation.mfp1(alist, g)

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

