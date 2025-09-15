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
from cosmology import *

NSIDE=32
SOURCE_MODEL=-2


# --- SIMPLE MAP ---
def create_average_flux_map(source_bias, dndr, zs, source_profile):
    npix = len(source_bias[0])
    nside = hp.npix2nside(npix)
    s0, sind = source_profile

    # The exposure is in km^2, so we need to divide by this
    mpc_in_km = 3.086e19
    at = 1 / (mpc_in_km)**2

    sangle = hp.nside2pixarea(nside)

    z = zs.reshape((len(zs), 1)) # Converting to proper column vector for ease of calculations
    dndr = dndr.reshape((len(z), 1))

    dz = np.gradient(z, axis=0)

    dl = z2dprop(z) * (1+z)
    dv = sangle * dl ** 2 / (1+z) * dz * C / H0 / sqrtH(z) / (1+z) # TODO dl->da?

    full_n = np.outer(dndr / s0 * (1+z) / (4 * np.pi * dl ** 2), at) # Mean amount of rays from a single source
    p = full_n # This is intentionally written in a confusing way, to show the similarity to the monte carlo function
    raycount = s0 * (1+z) ** sind * source_bias * dv * p
    
    hitmap = np.sum(raycount, axis=0)
    
    return hitmap

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

def plot_average_map(nside, b, zs, e0min, e0max, source_model, source_profile, exposure_name):    
    dndr = propagation.calc_cosmic_ray_rate_density(e0min, e0max, zs, source_model)

    at = exposure.create_exposure_map(nside, exposure_name)

    hitmap = create_average_map(b, dndr, zs, at, source_profile)

    ptitle = f"{e0min} < E0 < {e0max}; s0={source_profile[0]}; $\gamma$={source_model};"

    # g2sg = hp.Rotator(rot=(137.37, 0, 83.68))
    # hitmap = g2sg.rotate_map_pixel(hitmap)

    # hitmap = hp.sphtfunc.smoothing(hitmap, sigma=20 * np.pi / 180)
    # hitmap /= np.sum(hitmap) / 1e4
    hp.mollview(hitmap, title=ptitle)
    # hp.mollview(hitmap, title=ptitle, min=0, max=1.9)

    return np.sum(hitmap)


# --- MONTE CARLO - IN PARTS ---
def create_random_source_map(source_bias, z, source_profile, rng):
    npix = len(source_bias[0])
    nside = hp.npix2nside(npix)
    sangle = hp.nside2pixarea(nside)
    s0, sind = source_profile

    z = z.reshape((len(z), 1)) # Converting to proper column vector for ease of calculations
    dz = np.gradient(z, axis=0)
    
    dprop = z2dprop(z)
    da = dprop / (1+z)
    dv = sangle * da ** 2 * dz * C / H0 / sqrtH(z) / (1+z)

    mean_source = source_bias * dv * s0 * (1+z) ** sind

    sources = rng.poisson(mean_source)

    return sources

def generate_rays_from_sources(sources, dndr, zs, exposure, source_profile, rng, smear=0):
    npix = len(sources[0])
    s0, _ = source_profile

    # The exposure is in km^2, so we need to divide by this
    mpc_in_km = 3.086e19
    at = exposure / (mpc_in_km)**2 

    z = zs.reshape((len(zs), 1)) # Converting to proper column vector for ease of calculations
    dndr = dndr.reshape((len(z), 1))

    dprop = z2dprop(z)
    dl = dprop * (1+z)

    persource_flux = dndr / s0 * (1+z) / (4 * np.pi * dl ** 2) # TODO calculate this once and for all... there is a more efficient way of doing this whole thing

    full_flux = persource_flux * sources

    flux = np.sum(full_flux, axis=0)

    if smear != 0:
        # That 1e8 thing is to avoid overflow 
        flux = hp.smoothing(flux / 1e8, smear * np.pi/180) * 1e8
        flux = np.maximum(flux, 0)

    means = flux * at

    # Cosmic variance (var = expected var * (1 + this))
    # combined = np.outer(persource_flux, at)
    # oh = np.mean(combined, axis=0)
    # hp.mollview(oh)
    # plt.show()
    return rng.poisson(means)


# These two functions are just to do generate_rays_from_sources, read it to understand the logic
def fast_flux_factor(zs, source_profile, dndr, exposure):
    # The exposure is in km^2, so we need to divide by this
    mpc_in_km = 3.086e19
    at = exposure / (mpc_in_km)**2 
    
    z = zs.reshape((len(zs), 1)) # Converting to proper column vector for ease of calculations
    dndr = dndr.reshape((len(z), 1))

    dl = z2dprop(z) * (1+z)
    s0, _ = source_profile

    persource_flux = dndr / s0 * (1+z) / (4 * np.pi * dl ** 2)

    return persource_flux * at

def fast_generate_rays_from_sources(sources, flux_factor, rng):
    means = np.sum(sources * flux_factor, axis=0)
    return rng.poisson(means)


# --- IO ---
def save_hitmap(hitmap, source_model, source_profile, bratio, name, idx, path):
    gind = source_model
    s0 = source_profile[0]
    logs0 = int(np.log10(s0))

    full_name = f"cr_{name}_m{gind}_s{logs0}_b{bratio}_{idx}"
    full_path = path + "/" + full_name

    # Absolutely no way we get more than 10000 rays in one pixel
    hitmapint = hitmap.astype(np.uint16)
    np.save(full_path, hitmapint)


# --- MAIN ---
def randomize_and_save(b, zs, at, source_model, source_profile, rng, save_path, expname, bratio):
    nside = hp.get_nside(at)

    es = np.linspace(2e19, 8e19, 7)
    es[-1] = 2e22

    dndrs = []
    for i in range(len(es) - 1):
        dndrs.append(propagation.calc_cosmic_ray_rate_density(es[i], es[i+1], zs, source_model))

    dndrs_pro = []
    for i in range(len(es) - 1):
        dndrs_pro.append(propagation.calc_cosmic_ray_rate_density(es[i], es[i+1], zs, 0))


    flux_factors = [fast_flux_factor(zs, source_profile, dndr, at) for dndr in dndrs]
    flux_factors_pro = [fast_flux_factor(zs, source_profile, dndr, at) for dndr in dndrs_pro]

    # mf = analysis.BigMatchedFilterTest(create_average_map(b, sum(dndrs), zs, at, source_profile))
    # mf.save(f"cr_output/patterns/mf_{expname}_b{bratio}_e2_nuc")
    # mf = analysis.BigMatchedFilterTest(create_average_map(b, sum(dndrs_pro), zs, at, source_profile))
    # mf.save(f"cr_output/patterns/mf_{expname}_b{bratio}_e2_pro")
    # mf = analysis.BigMatchedFilterTest(create_average_map(b, sum(dndrs[2:]), zs, at, source_profile))
    # mf.save(f"cr_output/patterns/mf_{expname}_b{bratio}_e4_nuc")
    # mf = analysis.BigMatchedFilterTest(create_average_map(b, sum(dndrs_pro[2:]), zs, at, source_profile))
    # mf.save(f"cr_output/patterns/mf_{expname}_b{bratio}_e4_pro")
    # return

    for i in tqdm.tqdm(range(10000)):
        sources = create_random_source_map(b, zs, source_profile, rng)
        # sources_iso = create_random_source_map(iso_b, zs, source_profile, rng)

        for j, dndr in enumerate(dndrs):
            # hitmap = generate_rays_from_sources(sources, dndr, zs, at, source_profile, rng, 0)
            # hitmap_pro = generate_rays_from_sources(sources, dndrs_pro[j], zs, at, source_profile, rng, 0)
            # hitmap_iso = generate_rays_from_sources(sources_iso, dndr, zs, at, source_profile, rng, 0)
            # hitmap_iso_pro = generate_rays_from_sources(sources_iso, dndrs_pro[j], zs, at, source_profile, rng, 0)

            hitmap = fast_generate_rays_from_sources(sources, flux_factors[j], rng)
            hitmap_pro = fast_generate_rays_from_sources(sources, flux_factors_pro[j], rng)
            
            if None != save_path:
                save_hitmap(hitmap, -2, source_profile, bratio, f"{expname}_e{int(es[j] / 1e19)}", i, save_path)
                save_hitmap(hitmap_pro, 0, source_profile, bratio, f"{expname}_e{int(es[j] / 1e19)}", i, save_path)
                # save_hitmap(hitmap_iso, -2, source_profile, i, f"auger10I_e{int(es[j] / 1e19)}", save_path)
                # save_hitmap(hitmap_iso_pro, 0, source_profile, i, f"auger10I_e{int(es[j] / 1e19)}", save_path)

    return



# --- MAIN ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nside", "-n", help="nside for the healpix map", default=NSIDE)
    parser.add_argument("--source-model", "-s", help="model name for source emission", type=int, default=SOURCE_MODEL)
    parser.add_argument("--exposure", "-e", choices=['isotropic', 'auger', 'auger10', 'ta', '2022'], help="sky exposure pattern to use (default: isotropic)", default='isotropic')
    parser.add_argument("--source-density", "-sd", help="log10 source density (Mpc^-3)", type=float, default=-2.0)
    parser.add_argument("--source-evolution", "-se", help="source evolution index", type=float, default=0.0)
    parser.add_argument("--bias", "-b", choices=['iso', 'neutral', 'high'], help="source distribution bias to 2MRS", default='neutral')
    parser.add_argument("--output-directory", "-o", help="output path to save results in", default=None)
    return parser.parse_args()

def main():

    args = parse_args()
    rng = np.random.default_rng()

    zs = np.linspace(0, 0.4, 401)[1:] # Important: resolution need to be better than the bias map voxel size

    bratio = {"iso": 0, "neutral": 1, "high": 2/1.25}[args.bias]

    # 200 seems more or less the limit where the avg angular separation under 10 deg
    b = lss.create_source_bias_map_mrsl("MRS/catalog/2mrs_1175_done.dat", "MRS/CORRECTIONS/nearby.txt", args.nside, zs, 0.5, 200, bias_ratio=bratio)
    
    source_profile = (np.power(10, args.source_density), args.source_evolution)

    source_model = args.source_model
    at = exposure.create_exposure_map(args.nside, args.exposure)

    # print(plot_average_map(NSIDE, b, zs, 2e19, 2e21, -2, source_profile, "auger10"))
    # print(plot_average_map(NSIDE, b, zs, 2e19, 2e21, 0, source_profile, "auger10"))
    # for e0 in np.array([1.99526231e+19, 2.51188643e+19, 3.16227766e+19, 3.98107171e+19, 5.01187234e+19, 6.30957344e+19, 7.94328235e+19, 1.00000000e+20, 1.25892541e+20]):
    #     print(plot_average_map(NSIDE, b, zs, e0, 2e21, -2, source_profile, "auger10"))
    # plt.show()

    randomize_and_save(b, zs, at, source_model, source_profile, rng, args.output_directory, args.exposure, bratio)
    return

if __name__ == "__main__":
    main()
