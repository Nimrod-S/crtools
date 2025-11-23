from decimal import Decimal
import tqdm
import itertools
from crpropa import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from cosmology import *
import propagation

# Z for each A
# A=5 and A=8 are made up, A=1 is just hydrogen
STABLE_ISOTOPES = {2: 1, 3: 2, 4: 2, 6: 3, 7: 4, 9: 4, 10: 5, 11: 5, 12: 6, 13: 6, 14: 7, 15: 7, 16: 8, 17: 8, 18: 8, 19: 9, 20: 10, 21: 10, 22: 10, 23: 11, 24: 12, 25: 12, 26: 12, 27: 13, 28: 14, 29: 14, 30: 14, 31: 15, 32: 16, 33: 16, 34: 16, 36: 18, 35: 17, 37: 18, 38: 18, 40: 20, 39: 19, 41: 20, 42: 20, 43: 20, 44: 22, 45: 21, 46: 22, 47: 22, 48: 22, 49: 23, 50: 24, 51: 24, 52: 24, 53: 25, 54: 26, 55: 26, 56: 26, 1: 1, 5: 3, 8: 4}


def define_simulation():
    simulation = ModuleList()
    
    # Propagation type
    simulation.add(SimplePropagation()) # We don't care about magnetic fields so this one is good

    # Interactions
    # We're using all relevant interactions for nuclei because why not.
    for photobackground in [CMB, IRB_Gilmore12]: #TODO think about IRB
        simulation.add(ElectronPairProduction(photobackground()))
        simulation.add(PhotoDisintegration(photobackground()))
        simulation.add(PhotoPionProduction(photobackground()))
    simulation.add(NuclearDecay())

    # Cosmology
    # We are NOT adding FutureRedshift and AdiabaticCooling. Need to think about it. TODO
    simulation.add(Redshift())

    # Conditions
    #simulation.add(MinimumEnergy(1 * EeV)) # TODO change this

    return simulation


def define_observer():
    # Add a simple observer
    observer = Observer()

    observer.add(Observer1D()) # Just someone at x=0

    return observer

def define_ray(a, g, d_Mpc):
    e = 1 * GeV * a * g
    ray = Candidate(nucleusId(a, STABLE_ISOTOPES[a]), e, Vector3d(d_Mpc * Mpc, 0, 0))

    return ray

def define_source(d_Mpc):
    source = Source()
    source.add(SourceUniformHollowSphere(Vector3d(0, 0, 0), 50 * kpc, d_Mpc * Mpc))
    sc = SourceComposition(1e17 * eV, 1.3e18, 2)
    sc.add(nucleusId(56, 26), 1/15 / 56**3)
    sc.add(nucleusId(28, 14), 1/6 / 28**3)
    sc.add(nucleusId(14, 7), 1 / 14**3)
    source.add(sc)

    return source


def a_of_ray(r):
    return (r.getId() % 1000) / 10


def e_to_g(e, a):
    return e / (a * propagation.MP)

# A, g, d
# Distances are of the type that crpropa eats, therefore comoving distances (seen in documentation)
def generate_test_cases():
    energies = [2e19, 3e19, 4e19, 5e19, 6e19, 8e19, 1e20, 2e20]
    masses = [12, 14, 16, 28, 56] # C, N, O, Si, Fe
    distances = np.linspace(0, 1000)
    cases = itertools.product(masses, energies, distances)

    real_cases = []
    for c in cases:
        a, e, d = c[0], c[1], c[2]
        g = e_to_g(e, a)
        if propagation.lossfactor(a, g, d) * e < 1e19: 
            # Boring test
            continue
        real_cases.append((a, g, d))

    return real_cases

def present_results(results, on_g):

    
    for a in results.keys():
        # resultsgd = results[a]
        # gs = list(resultsgd.keys())
        # ds = list(resultsgd[gs[0]].keys())
        # table_sim = np.zeros((len(gs), len(ds)))
        # table_m1 = np.zeros((len(gs), len(ds)))
        # table_mc = np.zeros((len(gs), len(ds)))
        # for g in gs:
        #     gi = gs.index(g)
        #     resultsd = resultsgd[g]
        #     for d in ds:
        #         if d not in resultsd.keys():
        #             continue
        #         di = ds.index(d)
        #         mr, err, er, m1, m1f, mc, m2f, stop = resultsd[d]
        #         table_sim[gi][di] = mr
        #         table_m1[gi][di] = m1
        #         table_mc[gi][di] = mc

        # plt.figure()
        # plt.imshow(table_sim)
        
        # plt.figure()
        # plt.imshow(table_m1)

        # plt.figure()
        # plt.imshow(table_mc)

        # plt.show()

        # continue
        
        fig = plt.figure()
        fig.suptitle(f"A={a} old")
                            # plt.xlim(left=ds[0])

        if on_g:
            for i, g in enumerate(results[a].keys()):
                # ax = fig.add_subplot(3, 3, i + 1)
                # ax.set_title(f"$\gamma$={g:.2e}, E={g * propagation.MP * a}")
                # ax.set_xscale('log')
                # ax.set_yscale('log')
                # ax.set_ylim('')
                plt.yscale("log")
    
                ds = np.array(list(results[a][g].keys()))
                ds.sort()
                res = [results[a][g][d] for d in ds]
                mr, errs, er, m1, m1f, mc, m2f, stop = zip(*res)

                mr = np.array(mr)
                errs = np.array(errs)
                er = np.array(er)
                m1 = np.array(m1)
                m1f = np.array(m1f)
                mc = np.array(mc)
                m2f = np.array(m2f)

                #m3 = np.array([model3_ratio(a, g, d) for g in gs])

                # ax = plt.axes()
                # ax.set_xscale('log')
                # ax.set_yscale('log')
                # plt.errorbar(ds, mr * g * a * propagation.MP, errs * g * a * propagation.MP, color='royalblue')
                # plt.plot(ds, mr * g * a * propagation.MP, color='royalblue', label="mass ratio")
                # plt.plot(ds, m1 * g * a * propagation.MP, color='gray', label="model 1", linestyle='--')
                # plt.plot(ds, er * g * a * propagation.MP, color='dodgerblue', label="energy ratio")
                plt.plot(ds, mr * g * a * propagation.MP, color=f'C{i}', label="mass ratio", alpha=.6)
                plt.plot(ds, m1 * g * a * propagation.MP, color=f'C{i}', label="model 1", linestyle='--')
                plt.plot(ds, er * g * a * propagation.MP, color=f'C{i}', label="energy ratio")
                # plt.title(f"A={a}, E={'%.2E' % Decimal(g * 1e9 * a)} eV, g={'%.2E' % Decimal(g)}")
                # plt.show()
        else:
            for d in results[a].keys():
                gs = np.array(list(results[a][d].keys()))
                gs.sort()
                res = [results[a][d][g] for g in gs]
                mr, errs, er, m1, m1f, m2, m2f, stop = zip(*res)

                mr = np.array(mr)
                errs = np.array(errs)
                er = np.array(er)
                m1 = np.array(m1)
                m1f = np.array(m1f)
                m2 = np.array(m2)
                m2f = np.array(m2f)

                #m3 = np.array([model3_ratio(a, g, d) for g in gs])

                ax = plt.axes()
                ax.set_xscale('log')
                ax.set_yscale('log')
                #plt.errorbar(gs, mr, errs, color='blue')
                plt.plot(gs, mr, color='blue')
                plt.plot(gs, m1, color='red')
                # plt.plot(gs, m1f, color='orange')
                plt.plot(gs, m2, color='green')
                #plt.plot(gs, m2f, color='black')
                plt.plot(gs, er, color='gray')
                plt.xlim(left=gs[0])
                plt.title(f"A={a}, d={d} Mpc")
                plt.show()

                # ax = plt.axes()
                # ax.set_xscale('log')
                # ax.set_yscale('log')
                # plt.plot(gs, m1 / mr, color='red')
                # plt.plot(gs, m1f / mr, color='orange')
                # plt.plot(gs, m2 / mr, color='green')
                # plt.plot(gs, m2f / mr, color='black')
                # plt.plot(gs, er / mr, color='gray')
                # plt.show()
        plt.legend()

    plt.show()


            

def do_tests(on_g = False):
    test_cases = generate_test_cases()

    simulation = define_simulation()
    observer = define_observer()
    simulation.add(observer)

    results = {}
    
    for test in tqdm.tqdm(test_cases):
        #print(f"testing: A={test[0]}, g={test[1]}, d={test[2]} Mpc")
        rays = [define_ray(*test) for i in range(1000)]

        energy_ratios = []
        mass_ratios = []
        stops = []
    
        for ray in rays:
            simulation.run(ray)
            energy_ratios.append(ray.current.getEnergy() / ray.source.getEnergy())
            mass_ratios.append(a_of_ray(ray.current) / a_of_ray(ray.source))
            stops.append(ray.current.getPosition())

        counts, bins = np.histogram(np.array(mass_ratios) * test[0], np.linspace(0.5, test[0] + 0.5, test[0] + 1))
        #plt.stairs(counts, bins)
        #plt.show()
        # counts, bins = np.histogram(energy_ratios)
        # plt.stairs(counts, bins)
        # counts, bins = np.histogram(mass_ratios)
        # plt.stairs(counts, bins)
        # plt.show()

        a, g, d = test
        if on_g:
            if a not in results.keys():
                results[a] = {}
            if g not in results[a].keys():
                results[a][g] = {}
            results[a][g][d] = (np.mean(mass_ratios), np.std(mass_ratios), np.mean(energy_ratios), propagation.lossfactor_comovingd(*test), 0, propagation.lossfactor_comovingd(*test), 0, np.mean(stops))

        else:
            if a not in results.keys():
                results[a] = {}
            if d not in results[a].keys():
                results[a][d] = {}
            results[a][d][g] = (np.mean(mass_ratios), np.std(mass_ratios), np.mean(energy_ratios), propagation.lossfactor_comovingd(*test), 0, propagation.lossfactor_comovingd(*test), 0, np.mean(stops))

    present_results(results, on_g)


def max_len_tests():

    simulation = define_simulation()
    observer = define_observer()
    simulation.add(observer)

    a_list = [14, 28, 56]
    d_list = np.logspace(-1, 3, 16)
    e_list = [4e19, 6e19, 1e20, 3e20]

    for a in a_list:
        results = {}
        for e in tqdm.tqdm(e_list):
            g = e_to_g(e, a)
            dres = []
            for d in tqdm.tqdm(d_list):
                rays = [define_ray(a, g, d) for i in range(10000)]
                es = []
        
                for ray in rays:
                    simulation.run(ray)
                    #energy_ratios.append(ray.current.getEnergy() / ray.source.getEnergy())
                    #mass_ratios.append(a_of_ray(ray.current) / a_of_ray(ray.source))
                    #stops.append(ray.current.getPosition())
                    es.append(ray.current.getEnergy() / eV)
                if np.mean(es) / e < 1 / a:
                    break
                dres.append(np.mean(es))
            results[e] = dres
            plt.loglog(d_list[:len(dres)], dres, label=f'g={g}')
        plt.title(f"A={a}")
        plt.legend()
        plt.show()
    return


    results = {}

    d = 1e1
    a = 16
    e0 = 6e19
    
    # If it was exactly e0 at the source
    smallest_g = e_to_g(e0, a)
    gs = np.logspace(np.log10(smallest_g), 10, 16)

    finales = []

    for g in tqdm.tqdm(gs):
    
        rays = [define_ray(a, g, d) for i in range(10000)]

        energy_ratios = []
        mass_ratios = []
        stops = []

        es = []
        
        for ray in rays:
            simulation.run(ray)
            #energy_ratios.append(ray.current.getEnergy() / ray.source.getEnergy())
            #mass_ratios.append(a_of_ray(ray.current) / a_of_ray(ray.source))
            #stops.append(ray.current.getPosition())
            es.append(ray.current.getEnergy() / eV) 
        finales.append(np.mean(es))
    
    plt.loglog(gs, finales)
    plt.show()


def do_other_tests(a):

    e1s = np.geomspace(2e19, 2e20)
    ds = np.geomspace(1, 5000)
    e0s = np.array([1e20, 8e19, 6e19, 4e19, 2e19])
    e0s = np.array([10 ** 19.3, 10 ** 19.4, 10 ** 19.5, 10 ** 19.6, 10 ** 19.7][::-1])

    threshd = np.zeros((len(e0s), len(e1s)))

    simulation = define_simulation()
    observer = define_observer()
    simulation.add(observer)

    for ie, e1 in tqdm.tqdm(list(enumerate(e1s))):
        e0found = 0
        dindex = 0
        while e0s[e0found] > e1:
            e0found += 1

        while e0found != len(e0s):
            d = ds[dindex]
            dindex += 1
            if dindex == len(ds):
                break
        
            rays = [define_ray(a, e_to_g(e1, a), d) for _ in range(1000)]

            energy_reached = []
            energy_ratios = []
            mass_ratios = []
            stops = []
    
            for ray in rays:
                simulation.run(ray)
                energy_reached.append(ray.current.getEnergy() / eV)
                energy_ratios.append(ray.current.getEnergy() / ray.source.getEnergy())
                mass_ratios.append(a_of_ray(ray.current) / a_of_ray(ray.source))
                stops.append(ray.current.getPosition())
            
            e0 = np.mean(energy_reached) # TODO unclear if this is good!
            
            if e0 < e0s[e0found]:
                threshd[e0found][ie] = d
                e0found += 1
    

    np.save(f"thelines{a}", threshd)
    # for thresh in threshd:
        # plt.loglog(e1s, thresh)
    # plt.show()           



def main():
    # do_tests(on_g=True)
    # do_other_tests(56)
    # do_other_tests(12)
    # do_other_tests(16)
    # do_other_tests(28)
    do_other_tests(14)
    # max_len_tests()

    return



if "__main__" == __name__:
    main()
