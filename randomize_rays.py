import math
import argparse
import tqdm
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import healpy as hp

import analysis
import exposure
import lss
import propagation
import gmf
#import crpropa_gmf
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

def sum_of_uniforms(counts, loc, scale):
    tots = np.zeros(counts.shape)

    i = 0
    while True:
        idx = np.where(counts > i)
        n = len(counts[idx])
        if n == 0:
            break
        tots[idx] += sp.stats.uniform(loc=loc, scale=scale).rvs(size=n)

    """
    isnz = sources > 0
    snz = sources[isnz]
    lumf = sp.stats.irwinhall(snz, loc=2/11, scale=18/11).rvs()
    sources.astype(np.float64)[isnz] *= lumf
        """
    return tots

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

def save_meanmap(hitmap, source_model, source_profile, name, path):
    gind = source_model
    s0 = source_profile[0]
    bratio = source_profile[2]
    logs0 = int(np.log10(s0))

    full_name = f"mean_{name}_m{gind}_s{logs0}_b{bratio}"
    full_path = path + "/" + full_name

    np.save(full_path, hitmap)

    return

def old_lens_create(es, nside, output_directory):
    rig = np.linspace(10 ** 17.5, 10 ** 19.4, 3000)
    rsa = [propagation.get_r_dist(e, rig, -2) for e in es]
    rsad = [rsa[i+1] - rsa[i] for i in range(len(es)-1)]
    raa = [np.sum(rig * rs) / np.sum(rs) for rs in rsad]
    rsp = [e for e in es]
    rap = [(rsp[i+1] + rsp[i]) * 0.5 for i in range(len(es) -1)]
    lens_n = [[crpropa_gmf.proplens("UF23", r, nside, modeltype=m) for r in raa] for m in range(8)]
    lens_p = [[crpropa_gmf.proplens("UF23", r, nside, modeltype=m) for r in rap] for m in range(8)]

    # for mag in ["UF23", "JF12", "KST24"]:
    for m in range(8):
        for j in range(len(raa)):
            lnn = lens_n[m][j]
            lpp = lens_p[m][j]
            np.save(output_directory+f"/lens/l_nuc_UF23.{m}_e{(es[j] / 1e19):.3g}", lnn)
            np.save(output_directory+f"/lens/l_pro_UF23.{m}_e{(es[j] / 1e19):.3g}", lpp)
    
    return


# --- MAIN ---
def randomize_and_save(nside, s, source_profile, zs, expname, output_directory, gmf=False, lum=False, save=True):

    at = {exp: exposure.create_exposure_map(nside, exp) for exp in expname}
    
    es = np.load(output_directory+"/flux/energies_v2.npy")
    dndrs = np.load(output_directory+"/flux/flux_nuc_v2.npy")
    dndrs_pro = np.load(output_directory+"/flux/flux_pro_v2.npy")

    basename = "v2"
    if lum:
        basename += "_L"

    n_nuc = {
        exp: [create_mean_persousrce_count_map(zs, source_profile, dndr, at[exp]) for dndr in dndrs]
        for exp in expname
        }
    n_pro = {
        exp: [create_mean_persousrce_count_map(zs, source_profile, dndr, at[exp]) for dndr in dndrs_pro]
        for exp in expname
    }

    """
    for exp in expname:
        mean_n = []
        mean_p = []
        for j in range(len(dndrs)):

            mean_n.append(np.sum(s * n_nuc[exp][j], axis=0))
            mean_p.append(np.sum(s * n_pro[exp][j], axis=0))

        mean_n = np.array(mean_n)
        mean_p = np.array(mean_p)
        save_meanmap(mean_n, -2, source_profile, f"v2", output_directory+f"/meanmaps/{exp}")
        save_meanmap(mean_p, 0, source_profile, f"v2", output_directory+f"/meanmaps/{exp}")
    return
    """


    lens_n = [[np.load(output_directory + f"/lens/l_nuc_UF23.{m}_e{(e / 1e19):.3g}.npy") for e in es[:-1]] for m in range(8)]
    lens_p = [[np.load(output_directory + f"/lens/l_pro_UF23.{m}_e{(e / 1e19):.3g}.npy") for e in es[:-1]] for m in range(8)]

    #for i in tqdm.tqdm(range(10000)):
    for i in range(10000):
        sources = sp.stats.poisson(s).rvs()

        if lum:
            sources = sum_of_uniforms(sources, 2/11, 18/11)

        for exp in expname:

            hitmaps_n = []
            hitmaps_p = []

            if gmf:
                hitmaps_n_mag = [[] for m in range(8)]
                hitmaps_p_mag = [[] for m in range(8)]

            for j in range(len(dndrs)):

                mean_n = np.sum(sources * n_nuc[exp][j], axis=0)
                mean_p = np.sum(sources * n_pro[exp][j], axis=0)

                hitmaps_n.append(sp.stats.poisson(mean_n).rvs())
                hitmaps_p.append(sp.stats.poisson(mean_p).rvs())

                if not gmf:
                    continue

                for m in range(8):
                    sources_n = sources[:, lens_n[m][j]].reshape((len(zs), len(at[exp])))
                    sources_p = sources[:, lens_p[m][j]].reshape((len(zs), len(at[exp])))

                    mean_n = np.sum(sources_n * n_nuc[exp][j], axis=0)
                    mean_p = np.sum(sources_p * n_pro[exp][j], axis=0)

                    hitmaps_n_mag[m].append(sp.stats.poisson(mean_n).rvs())
                    hitmaps_p_mag[m].append(sp.stats.poisson(mean_p).rvs())


            if save:
                if not gmf:
                    save_hitmap(np.array(hitmaps_n), -2, source_profile, basename, i, output_directory+f"/hitmaps/{exp}")
                    save_hitmap(np.array(hitmaps_p), 0, source_profile, basename, i, output_directory+f"/hitmaps/{exp}")
                if gmf:
                    for m in range(8):
                        save_hitmap(np.array(hitmaps_n_mag[m]), -2, source_profile, f"{basename}_U{m}", i, output_directory+f"/hitmaps/{exp}")
                        save_hitmap(np.array(hitmaps_p_mag[m]), 0, source_profile, f"{basename}_U{m}", i, output_directory+f"/hitmaps/{exp}")


                
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

    parser.add_argument("--gmf", help="whether to consider gmf", action="store_true")
    parser.add_argument("--lum", help="whether to consider variable luminosity", action="store_true")
    return parser.parse_args()

def main():

    args = parse_args()

    zs = np.linspace(0, 0.4, 401)[1:] # Important: resolution need to be better than the bias map voxel size

    bratio = {"iso": 0, "neutral": 1, "high": 1.7}[args.bias]

    b = np.load(args.output_directory + "/lss/lss_bias_v1.npy")

    for sd in [np.power(10.0, int(args.source_density))]:
        source_profile = (sd, args.source_evolution, bratio)
        s = create_mean_source_count_map(b, zs, source_profile)
        randomize_and_save(args.nside, s, source_profile, zs, ["auger", "2022"], args.output_directory, gmf=args.gmf, lum=args.lum)
        
    return

if __name__ == "__main__":
    main()
