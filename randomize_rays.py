import argparse
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import healpy as hp

import exposure
import lss
import propagation
import gmf
import crpropa_gmf
from cosmology import *

NSIDE=32


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


def create_mean_persource_flux(zs, source_profile, dndr):
    z = zs.reshape((len(zs), 1)) # Converting to proper column vector for ease of calculations
    dndr = dndr.reshape((len(z), 1))

    dl = z2dprop(z) * (1+z)
    s0, _, _ = source_profile

    persource_flux = dndr / s0 * (1+z) / (4 * np.pi * dl ** 2)
    return persource_flux

def create_rotation_smear_matrix(nside, npix, smear_th, smear_phi, v1, v2, v3):
    v_smeared = v1 * np.cos(smear_th)[:, np.newaxis] + v2 * (np.sin(smear_th) * np.cos(smear_phi))[:, np.newaxis] + v3 * (np.sin(smear_th) * np.sin(smear_phi))[:, np.newaxis]
    i_smeared = hp.vec2pix(nside, *v_smeared.transpose())
    return i_smeared

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

def save_smearmap(smearmap, source_model, name, path):
    gind = source_model

    full_name = f"smear_{name}_m{gind}"
    full_path = path + "/" + full_name

    np.save(full_path, smearmap)

# --- Secondary functions ---
def s_lens_create(es, nside, output_directory):
    rap = propagation.get_mean_r(es, 0)
    raa = propagation.get_mean_r(es, -2)
    lens_n = [[crpropa_gmf.proplens("UF23", r, nside, modeltype=m) for r in raa] for m in range(8)]
    lens_p = [[crpropa_gmf.proplens("UF23", r, nside, modeltype=m) for r in rap] for m in range(8)]

    for m in range(8):
        for j in range(len(raa)):
            lnn = lens_n[m][j]
            lpp = lens_p[m][j]
            np.save(output_directory+f"/lens/l_nuc_UF23.{m}_e{(es[j] / 1e19):.3g}", lnn)
            np.save(output_directory+f"/lens/l_pro_UF23.{m}_e{(es[j] / 1e19):.3g}", lpp)
    
    return

def s_smear_create(es, nside, output_directory, gmfmod):
    npix = hp.nside2npix(nside)

    rig_nuc = propagation.get_mean_r(es, -2)
    rig_pro = propagation.get_mean_r(es, 0)

    _, lats = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    if gmfmod != -1:
        def_factors_n = [crpropa_gmf.proplength("UF23", r, nside, modeltype=gmfmod) * np.abs(np.sin(lats * np.pi/180)) for r in rig_nuc]
        def_factors_p = [crpropa_gmf.proplength("UF23", r, nside, modeltype=gmfmod) * np.abs(np.sin(lats * np.pi/180)) for r in rig_pro]
        magname= f"_U{gmfmod}"
    else:
        def_factors_n = [np.ones(npix) for e in es[:-1]]
        def_factors_p = [np.ones(npix) for e in es[:-1]]
        magname = ""

    smear_map_n = [gmf.deflection_random(r, gmf.rmstd(lats)) * def_factors_n[i] for i, r in enumerate(rig_nuc)]
    smear_map_p = [gmf.deflection_random(r, gmf.rmstd(lats)) * def_factors_p[i] for i, r in enumerate(rig_pro)]

    save_smearmap(smear_map_n, -2, "v2"+magname, output_directory+f"/smearmaps/")
    save_smearmap(smear_map_p, 0, "v2"+magname, output_directory+f"/smearmaps/")

def s_meanmap_create(nside, s, source_profile, zs, expname, output_directory, gmfmod):

    npix = hp.nside2npix(nside)

    mpc_in_km = 3.086e19 # Since we want it in mpc units
    at = {exp: exposure.create_exposure_map(nside, exp) / (mpc_in_km)**2 for exp in expname}
    
    es = np.load(output_directory+"/flux/energies_v2.npy")
    dndrs = np.load(output_directory+"/flux/flux_nuc_v2.npy")
    dndrs_pro = np.load(output_directory+"/flux/flux_pro_v2.npy")

    n_nuc = [create_mean_persource_flux(zs, source_profile, dndr) for dndr in dndrs]
    n_pro = [create_mean_persource_flux(zs, source_profile, dndr) for dndr in dndrs_pro]
    
    if gmfmod != -1:
        lens_n = [np.load(output_directory + f"/lens/l_nuc_UF23.{gmfmod}_e{(e / 1e19):.3g}.npy") for e in es[:-1]]
        lens_p = [np.load(output_directory + f"/lens/l_pro_UF23.{gmfmod}_e{(e / 1e19):.3g}.npy") for e in es[:-1]]
        magname= f"_U{gmfmod}"
    else:
        lens_n = [np.arange(npix) for e in es[:-1]]
        lens_p = [np.arange(npix) for e in es[:-1]]
        magname = ""

    for exp in expname:
        mean_n = []
        mean_p = []
        for j in range(len(dndrs)):

            flux_n = np.sum(n_nuc[j] * s, axis=0)
            flux_p = np.sum(n_pro[j] * s, axis=0)
            
            if gmfmod != -1:
                flux_n = flux_n[lens_n[j]]
                flux_p = flux_p[lens_p[j]]

            mean_n.append(flux_n * at[exp])
            mean_p.append(flux_p * at[exp])

        mean_n = np.array(mean_n)
        mean_p = np.array(mean_p)
        save_meanmap(mean_n, -2, source_profile, f"v2{magname}", output_directory+f"/meanmaps/{exp}")
        save_meanmap(mean_p, 0, source_profile, f"v2{magname}", output_directory+f"/meanmaps/{exp}")
    return

# --- MAIN ---
def randomize_and_save_rays(nside, source_profile, zs, expname, output_directory, gmfmod=-1, lum=False, lenses=False, save=True):

    npix = hp.nside2npix(nside)

    mpc_in_km = 3.086e19 # Since we want it in mpc units
    at = {exp: exposure.create_exposure_map(nside, exp) / (mpc_in_km)**2 for exp in expname}
    
    es = np.load(output_directory+"/flux/energies_v2.npy")
    dndrs = np.load(output_directory+"/flux/flux_nuc_v2.npy")
    dndrs_pro = np.load(output_directory+"/flux/flux_pro_v2.npy")

    basename = "v2"
    if lum:
        basename += "_L"

    if lenses:
        s_smear_create(es, nside, output_directory, gmfmod)
        s_lens_create(es, nside, output_directory)

    n_nuc = [create_mean_persource_flux(zs, source_profile, dndr) for dndr in dndrs]
    n_pro = [create_mean_persource_flux(zs, source_profile, dndr) for dndr in dndrs_pro]

    if gmfmod != -1:
        lens_n = [np.load(output_directory + f"/lens/l_nuc_UF23.{gmfmod}_e{(e / 1e19):.3g}.npy") for e in es[:-1]]
        lens_p = [np.load(output_directory + f"/lens/l_pro_UF23.{gmfmod}_e{(e / 1e19):.3g}.npy") for e in es[:-1]]
        magname= f"_U{gmfmod}"
    else:
        lens_n = [np.arange(npix) for e in es[:-1]]
        lens_p = [np.arange(npix) for e in es[:-1]]
        magname = ""

    smear_map_n = np.load(output_directory + f"/smearmaps/smear_v2{magname}_m-2.npy")
    smear_map_n = np.where(np.isinf(smear_map_n) | np.isnan(smear_map_n), 2*np.pi, smear_map_n)
    smear_map_p = np.load(output_directory + f"/smearmaps/smear_v2{magname}_m0.npy")
    smear_map_p = np.where(np.isinf(smear_map_p) | np.isnan(smear_map_p), 2*np.pi, smear_map_p)

    th, ph = hp.pix2ang(nside, np.arange(npix))
    v1 = np.column_stack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))
    v2 = np.column_stack((-np.sin(ph), np.cos(ph), np.zeros(npix)))
    v3 = np.column_stack((np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)))

    s0 = source_profile[0]
    logs0 = int(np.log10(s0))
    source_path = output_directory + f"/sourcemaps/source_v1_s{logs0}_b{source_profile[2]}_"

    # for i in tqdm.tqdm(range(10000)):
    for i in range(10000):
        #sources = sp.stats.poisson(s).rvs()
        sources = np.load(source_path+f"{i}.npy")

        # Randomize deflections
        smear_phi = sp.stats.uniform(0, 2*np.pi).rvs(npix)
        smear_th_norm = sp.stats.rayleigh().rvs(npix) # Normalized to one radian

        if lum:
            sources = sum_of_uniforms(sources, 2/11, 18/11)

        for exp in expname:
            hitmaps_n = []
            hitmaps_p = []

            for j in range(len(dndrs)):
                source_flux_n = np.sum(n_nuc[j] * sources, axis=0)
                source_flux_p = np.sum(n_pro[j] * sources, axis=0)

                smearmat_n = create_rotation_smear_matrix(nside, npix, smear_th_norm * smear_map_n[j], smear_phi, v1, v2, v3)
                smearmat_p = create_rotation_smear_matrix(nside, npix, smear_th_norm * smear_map_p[j], smear_phi, v1, v2, v3)

                source_flux_n = source_flux_n[lens_n[j]]
                source_flux_p = source_flux_p[lens_p[j]]

                source_flux_smeared_n = np.zeros(npix)
                source_flux_smeared_p = np.zeros(npix)
                for ipix in range(npix):
                    source_flux_smeared_n[smearmat_n[ipix]] += source_flux_n[ipix]
                    source_flux_smeared_p[smearmat_p[ipix]] += source_flux_p[ipix]

                mean_n = source_flux_smeared_n * at[exp]
                mean_p = source_flux_smeared_p * at[exp]

                hitmaps_n.append(sp.stats.poisson(mean_n).rvs())
                hitmaps_p.append(sp.stats.poisson(mean_p).rvs())

            if save:
                save_hitmap(np.array(hitmaps_n), -2, source_profile, basename + magname, i, output_directory+f"/hitmaps/{exp}")
                save_hitmap(np.array(hitmaps_p), 0, source_profile, basename + magname, i, output_directory+f"/hitmaps/{exp}")


def randomize_and_save_sources(nside, s, source_profile, zs, output_directory):

    s0 = source_profile[0]
    bratio = source_profile[2]
    logs0 = int(np.log10(s0))

    # for i in tqdm.tqdm(range(10000)):
    for i in range(10000):
        sources = sp.stats.poisson(s).rvs()

        full_name = f"source_v1_s{logs0}_b{bratio}_{i}"
        full_path = output_directory + "/sourcemaps/" + full_name

        # Absolutely no way we get more than 10000 sources in one pixel
        sources = sources.astype(np.uint16)
        np.save(full_path, sources)

    return

# --- MAIN ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nside", "-n", help="nside for the healpix map", default=NSIDE)
    parser.add_argument("--exposure", "-e", choices=['isotropic', 'auger', 'auger10', 'ta', '2022', 'ideal'], help="sky exposure pattern to use (default: ideal)", default='ideal')
    parser.add_argument("--source-density", "-sd", help="log10 source density (Mpc^-3)", type=float, default=-2.0)
    parser.add_argument("--source-evolution", "-se", help="source evolution index", type=float, default=0.0) # NOT USED IN THE PAPER
    parser.add_argument("--bias", "-b", choices=['iso', 'neutral', 'high'], help="source distribution bias to 2MRS", default='neutral')
    parser.add_argument("--output-directory", "-o", help="output path to save results in", default=None)

    parser.add_argument("--gmf", help="what gmf model to use (default: -1, none)", default=-1, type=int)
    parser.add_argument("--lum", help="whether to consider variable luminosity", action="store_true") # NOT USED IN THE PAPER

    parser.add_argument("--lenses", help="create lenses in addition to randomizing rays", action="store_true")
    parser.add_argument("--justs", help="randomize sources (instead of rays)", action="store_true")
    return parser.parse_args()

def main():

    args = parse_args()

    zs = np.linspace(0, 0.4, 401)[1:]

    bratio = {"iso": 0, "neutral": 1, "high": 1.7}[args.bias]

    b = np.load(args.output_directory + "/lss/lss_bias_v1.npy")

    for sd in [np.power(10.0, int(args.source_density))]:
        source_profile = (sd, args.source_evolution, bratio)
        if args.justs:
            s = create_mean_source_count_map(b, zs, source_profile)
            randomize_and_save_sources(args.nside, s, source_profile, zs, args.output_directory, lum=args.lum)
        else:
            
            randomize_and_save_rays(args.nside, source_profile, zs, [args.exposure], args.output_directory, gmfmod=args.gmf, lum=args.lum, lenses=args.lenses)
            #s_meanmap_create(args.nside, s, source_profile, zs, [args.exposure], args.output_directory, gmfmod=args.gmf)        
    return

if __name__ == "__main__":
    main()
