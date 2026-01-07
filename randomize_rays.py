import math
import argparse
import tqdm
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import healpy as hp
import corner

import analysis
import exposure
import lss
import propagation
import gmf
import crpropa_gmf
from cosmology import *

NSIDE=32
SOURCE_MODEL=-2


# --- SIMPLE MAP ---
def create_average_map(source_bias, dndr, zs, exposure, source_profile):
    npix = len(exposure)
    nside = hp.npix2nside(npix)
    s0, sind = source_profile

    # The exposure is in km^2, so we need to divide by this
    mpc_in_km = 3.086e19
    at = exposure / (mpc_in_km)**2

    sangle = hp.nside2pixarea(nside)

    z = zs.reshape((len(zs), 1)) # Converting to proper column vector for ease of calculations
    dndr = dndr.reshape((len(z), 1))

    dz = np.gradient(z, axis=0)

    da = z2dprop(z) / (1+z)
    dl = da * (1+z)**2
    dv = sangle * da ** 2 * dz * C / H0 / sqrtH(z) / (1+z)

    full_n = np.outer(dndr / s0 * (1+z) / (4 * np.pi * dl ** 2), at) # Mean amount of rays from a single source
    p = full_n # This is intentionally written in a confusing way, to show the similarity to calc_poisson_mean
    raycount = s0 * (1+z) ** sind * source_bias * dv * p
    
    hitmap = np.sum(raycount, axis=0)
    
    return hitmap

# DEPRECATED
def plot_average_map(nside, b, zs, e0min, e0max, source_model, source_profile, exposure_name):    
    dndr = propagation.calc_cosmic_ray_rate_density(e0min, e0max, zs, source_model)

    at = exposure.create_exposure_map(nside, exposure_name)

    hitmap = create_average_map(b, dndr, zs, at, source_profile)

    ptitle = f"{e0min} < E0 < {e0max}; s0={source_profile[0]}; $\gamma$={source_model};"

    # g2sg = hp.Rotator(rot=(137.37, 0, 83.68))
    # hitmap = g2sg.rotate_map_pixel(hitmap)

    # hitmap = hp.sphtfunc.smoothing(hitmap, sigma=20 * np.pi / 180)
    # hitmap /= np.sum(hitmap) / 1e4
    mn = np.mean(hitmap)
    dl = hitmap / mn - 1
    hp.mollview(dl, title="", min=-0.56, max=1.47)
    # hp.mollview(hitmap, title=ptitle, min=0, max=1.9)

    return np.sum(hitmap)


# --- MONTE CARLO - IN PARTS ---
def create_mean_source_count_map(source_bias, z, source_profile):
    npix = len(source_bias[0])
    nside = hp.npix2nside(npix)
    sangle = hp.nside2pixarea(nside)
    s0, sind, bratio = source_profile

    source_bias = lss.apply_linear_bias(source_bias, bratio)

    z = z.reshape((len(z), 1)) # Converting to proper column vector for ease of calculations
    dz = np.gradient(z, axis=0)
    
    dprop = z2dprop(z)
    da = dprop / (1+z)
    dv = sangle * da ** 2 * dz * C / H0 / sqrtH(z) / (1+z)

    mean_source = source_bias * dv * s0 * (1+z) ** sind

    return mean_source

def create_mean_persousrce_count_map(zs, source_profile, dndr, exposure):
    # The exposure is in km^2, so we need to divide by this
    mpc_in_km = 3.086e19
    at = exposure / (mpc_in_km)**2
    
    z = zs.reshape((len(zs), 1)) # Converting to proper column vector for ease of calculations
    dndr = dndr.reshape((len(z), 1))

    dl = z2dprop(z) * (1+z)
    s0, _, _ = source_profile

    persource_flux = dndr / s0 * (1+z) / (4 * np.pi * dl ** 2)

    return persource_flux * at

# --- IO ---
def save_hitmap(hitmap, source_model, source_profile, name, idx, path):
    gind = source_model
    s0 = source_profile[0]
    bratio = source_profile[2]
    logs0 = int(np.log10(s0))

    full_name = f"cr_{name}_m{gind}_s{logs0}_b{bratio}_{idx}"
    full_path = path + "/" + full_name

    # Absolutely no way we get more than 10000 rays in one pixel
    hitmapint = hitmap.astype(np.uint16)
    np.save(full_path, hitmapint)


# --- MAIN ---
def secondary(b, zs, at, source_model, source_profile, save_path, expname, bratio):

    q = np.array([81.0, 65, 65, 49, 50, 30, 31, 23, 21, 17, 18, 10,  6,  6,  8,  4,  2, 0,  0,  1,  4,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])
    q /= np.sum(q)

    ln, lln, lp, llp = [], [], [], []

    nside = hp.get_nside(at)
    es = np.logspace(19.3, 21)
    dndrs = []
    for i in range(len(es) - 1):
        dndrs.append(propagation.calc_cosmic_ray_rate_density(es[i], es[i+1], zs, source_model))
    dndrs_pro = []
    for i in range(len(es) - 1):
        dndrs_pro.append(propagation.calc_cosmic_ray_rate_density(es[i], es[i+1], zs, 0))
    flux_factors = [fast_flux_factor(zs, source_profile, dndr, at) for dndr in dndrs]
    flux_factors_pro = [fast_flux_factor(zs, source_profile, dndr, at) for dndr in dndrs_pro]

    solid_spec = [np.sum(_mean_sources(b, zs, source_profile) * ff) for ff in flux_factors]
    solid_spec /= np.sum(solid_spec)
    solid_specp = [np.sum(_mean_sources(b, zs, source_profile) * ff) for ff in flux_factors_pro]
    solid_specp /= np.sum(solid_specp)
    nq = sp.stats.kstest(q, solid_spec).statistic
    pq = sp.stats.kstest(q, solid_specp).statistic
    print(nq, pq)

    mfnuc = analysis.BigMatchedFilterTest.load(f"cr_output/patterns/mf_{expname}_b1_e2_nuc")
    mfpro = analysis.BigMatchedFilterTest.load(f"cr_output/patterns/mf_{expname}_b1_e2_pro")
    
    nuc_vs_nuc = []
    nuc_vs_pro = []
    pro_vs_nuc = []
    pro_vs_pro = []

    for i in tqdm.tqdm(range(1000)):
        sources = create_random_source_map(b, zs, source_profile)
        # sources_iso = create_random_source_map(iso_b, zs, source_profile)

        real = []
        realp = []
        hitmap = np.zeros(len(at))
        hitmap_pro = np.zeros(len(at))
        for j, dndr in enumerate(dndrs):
            # hitmap = generate_rays_from_sources(sources, dndr, zs, at, source_profile, 0)
            # hitmap_pro = generate_rays_from_sources(sources, dndrs_pro[j], zs, at, source_profile, 0)
            # hitmap_iso = generate_rays_from_sources(sources_iso, dndr, zs, at, source_profile, 0)
            # hitmap_iso_pro = generate_rays_from_sources(sources_iso, dndrs_pro[j], zs, at, source_profile, 0)

            hitmap += fast_generate_rays_from_sources(sources, flux_factors[j])
            hitmap_pro += fast_generate_rays_from_sources(sources, flux_factors_pro[j])

            # real.append(np.sum(sources * flux_factors[j]))
            # realp.append(np.sum(sources * flux_factors_pro[j]))

        nuc_vs_nuc.append(mfnuc.test(hitmap))
        nuc_vs_pro.append(mfpro.test(hitmap))
        pro_vs_nuc.append(mfnuc.test(hitmap_pro))
        pro_vs_pro.append(mfpro.test(hitmap_pro))
        # real = np.array(real)
        # realp = np.array(realp)
        # vals = sp.stats.poisson(real).rvs()
        # valsp = sp.stats.poisson(realp).rvs()
        # real /= np.sum(real)
        # realp /= np.sum(realp)
        # vals /= np.sum(vals)
        # valsp /= np.sum(valsp)
        # plt.figure()
        # plt.stairs(q, edges=es, linewidth=3, color="gold")
        # plt.stairs(solid_spec, edges=es, linewidth=2, color="black")
        # plt.stairs(vals, edges=es, linewidth=2, color="C0")
        # plt.stairs(real, edges=es, linewidth=1, color="C0", linestyle="--")
        # plt.xscale("log")
        # plt.xlim(right=2e20)
        # plt.figure()
        # plt.stairs(q, edges=es, linewidth=3, color="gold")
        # plt.stairs(solid_specp, edges=es, linewidth=2, color="black")
        # plt.stairs(valsp, edges=es, linewidth=2, color="C3")
        # plt.stairs(realp, edges=es, linewidth=1, color="C3", linestyle="--")
        # plt.xscale("log")
        # plt.xlim(right=2e20)
        # plt.show()

        # n1 = sp.stats.kstest(vals, solid_spec).statistic
        # p1 = sp.stats.kstest(valsp, solid_specp).statistic
        # print(n1, p1)
        
        # n1, n2, p1, p2 = np.linalg.norm(vals - q), np.linalg.norm(real - q), np.linalg.norm(valsp - q), np.linalg.norm(realp - q)
        # print(n1, n2, p1, p2)
        # ln.append(n1)
        # lln.append(n2)
        # lp.append(p1)
        # llp.append(p2)


    nuc_vs_nuc = np.array(nuc_vs_nuc)
    nuc_vs_pro = np.array(nuc_vs_pro)
    pro_vs_nuc = np.array(pro_vs_nuc)
    pro_vs_pro = np.array(pro_vs_pro)
    np.save("BLAH1", nuc_vs_nuc)
    np.save("BLAH2", nuc_vs_pro)
    np.save("BLAH3", pro_vs_nuc)
    np.save("BLAH4", pro_vs_pro)


    # plt.figure()
    # plt.hist(ln, color="C0", linewidth=3, linestyle="-", histtype='step')
    # plt.vlines([nq], 0, 20, color='black')
    # # plt.hist(lln, bins=bi, color="C0", linewidth=3, linestyle="--", histtype='step')
    # plt.figure()
    # plt.hist(lp, color="C3", linewidth=3, linestyle="-", histtype='step')
    # plt.vlines([pq], 0, 20, color='black')
    # # plt.hist(llp, bins=bi, color="C3", linewidth=3, linestyle="--", histtype='step')
    # plt.show()

    return


def randomize_and_save(nside, s, source_profile, zs, expname, output_directory, save=True):

    at = {exp: exposure.create_exposure_map(nside, exp) for exp in expname}
    
    es = np.load(output_directory+"/flux/energies_v1.npy")
    dndrs = np.load(output_directory+"/flux/flux_nuc_v1.npy")
    dndrs_pro = np.load(output_directory+"/flux/flux_pro_v1.npy")

    n_nuc = {
        exp: [create_mean_persousrce_count_map(zs, source_profile, dndr, at[exp]) for dndr in dndrs]
        for exp in expname
        }
    n_pro = {
        exp: [create_mean_persousrce_count_map(zs, source_profile, dndr, at[exp]) for dndr in dndrs_pro]
        for exp in expname
    }

    rig = np.linspace(10 ** 17.5, 10 ** 19.4, 3000)
    rsa = [propagation.get_r_dist(e, rig, -2) for e in es]
    rsad = [rsa[i+1] - rsa[i] for i in range(len(es)-1)]
    raa = [np.sum(rig * rs) / np.sum(rs) for rs in rsad]
    rsp = [e for e in es]
    rap = [(rsp[i+1] + rsp[i]) * 0.5 for i in range(len(es) -1)]
    lens_n = {name: [crpropa_gmf.proplens(name, r, nside) for r in raa] for name in ["UF23", "JF12", "KST24"]}
    lens_p = {name: [crpropa_gmf.proplens(name, r, nside) for r in rap] for name in ["UF23", "JF12", "KST24"]}

    for i in tqdm.tqdm(range(10000)):
        sources = sp.stats.poisson(s).rvs()
        
        # isnz = sources > 0
        # snz = sources[isnz]
        # lumf = sp.stats.irwinhall(snz, loc=2/11, scale=18/11).rvs()
        # sources.astype(np.float64)[isnz] *= lumf
        # lumf = sp.stats.irwinhall(sources + 1, loc=2/11, scale=18/11).rvs()
        # sources = sources.astype(np.float64) * lumf

        for exp in expname:
            for j in range(len(dndrs)):

                sources_n = sources[:, lens_n["KST24"][j]].reshape((len(zs), len(at[exp])))
                sources_p = sources[:, lens_p["KST24"][j]].reshape((len(zs), len(at[exp])))

                mean_n = np.sum(sources_n * n_nuc[exp][j], axis=0)
                mean_p = np.sum(sources_p * n_pro[exp][j], axis=0)

                hitmap = sp.stats.poisson(mean_n).rvs()
                hitmap_pro = sp.stats.poisson(mean_p).rvs()

                if save:
                    save_hitmap(hitmap, -2, source_profile, f"{exp}_e{int(es[j] / 1e19)}_KST24", i, output_directory+"/hitmaps")
                    save_hitmap(hitmap_pro, 0, source_profile, f"{exp}_e{int(es[j] / 1e19)}_KST24", i, output_directory+"/hitmaps")


    return

# --- MAIN ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nside", "-n", help="nside for the healpix map", default=NSIDE)
    parser.add_argument("--exposure", "-e", choices=['isotropic', 'auger', 'auger10', 'ta', '2022'], help="sky exposure pattern to use (default: isotropic)", default='isotropic')
    parser.add_argument("--source-density", "-sd", help="log10 source density (Mpc^-3)", type=float, default=-2.0)
    parser.add_argument("--source-evolution", "-se", help="source evolution index", type=float, default=0.0)
    parser.add_argument("--bias", "-b", choices=['iso', 'neutral', 'high'], help="source distribution bias to 2MRS", default='neutral')
    parser.add_argument("--output-directory", "-o", help="output path to save results in", default=None)
    return parser.parse_args()

def main():

    args = parse_args()

    zs = np.linspace(0, 0.4, 401)[1:] # Important: resolution need to be better than the bias map voxel size

    bratio = {"iso": 0, "neutral": 1, "high": 1.7}[args.bias]

    b = np.load(args.output_directory + "/lss/lss_bias_v1.npy")

    for sd in [1e-2, 1e-3, 1e-4]:
        source_profile = (sd, args.source_evolution, bratio)
        s = create_mean_source_count_map(b, zs, source_profile)
        randomize_and_save(args.nside, s, source_profile, zs, ["auger10", "isotropic", "2022"], args.output_directory)
        
    return

if __name__ == "__main__":
    main()
