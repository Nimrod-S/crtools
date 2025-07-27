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
    p = full_n # This is intentionally written in a confusing way, to show the similarity to calc_poisson_mean
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

    dl = z2dprop(z) * (1+z)
    dv = sangle * dl ** 2 / (1+z) * dz * C / H0 / sqrtH(z) / (1+z) # TODO dl->da?

    full_n = np.outer(dndr / s0 * (1+z) / (4 * np.pi * dl ** 2), at) # Mean amount of rays from a single source
    p = full_n # This is intentionally written in a confusing way, to show the similarity to calc_poisson_mean
    raycount = s0 * (1+z) ** sind * source_bias * dv * p
    
    hitmap = np.sum(raycount, axis=0)
    
    return hitmap

def plot_average_map(nside, b, zs, e0min, e0max, source_model, source_profile, exposure_name):    
    dndr = propagation.calc_cosmic_ray_rate_density(e0min, e0max, zs, source_model, 0, FUCKYOU=True)

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

# --- MONTE CARLO - FULL ---
def calc_poisson_mean(source_bias, dndr, zs, sangle, exposure, irays, source_profile):
    s0, sind = source_profile

    s = 0
    dzs = np.gradient(zs)

    # The exposure is in km^2, so we need to divide by this
    mpc_in_km = 3.086e19
    at = exposure / (mpc_in_km)**2 

    for i, z in enumerate(zs):
        b = source_bias[i]
        n = dndr[i]

        dprop = z2dprop(z)
        dl = dprop * (1+z)
        #dl = z * C / H0# TODO
        # TODO dl->da?
        dv = sangle * dl ** 2 / (1+z) * dzs[i] * C / H0 / sqrtH(z) / (1+z)

        if n == 0:
            p = 0
        else:
            full_n = n * at * (1+z) / s0 / (4 * np.pi * dl ** 2) # Mean amount of rays from a single source
            # Doing the poisson distribution with this log trick to make me and numpy happy (smaller numbers)
            p = np.exp(irays * np.log(full_n) - full_n) / math.factorial(irays)
        
        s += s0 * (1+z) ** sind * b * p * dv
    
    return s

def create_poisson_means_map(source_bias, dndr, zs, exposure, source_profile, mean_threshold=0.001):
    #TODO choice of threshold
    npix = len(exposure)
    nside = hp.npix2nside(npix)

    means_map = [[] for _ in range(npix)] # List of lists per i, starting at i=1

    for ipix in range(npix):
        for irays in range(1, 10000): # Very big number we won't reach
            si = calc_poisson_mean(source_bias.transpose()[ipix], dndr, zs, hp.nside2pixarea(nside), exposure[ipix], irays, source_profile)
            if si * irays < mean_threshold:
                break
            means_map[ipix].append(si)

    return means_map
    
def generate_rays(poisson_means, rng):
    npix = len(poisson_means)
    nside = hp.npix2nside(npix)

    hitmap = np.zeros(npix, dtype=int)

    for ipix in range(npix):
        for i, si in enumerate(poisson_means[ipix]):
            iray = i + 1 # See create_poisson_means_map, this is the lowest ray count
            n = rng.poisson(si)
            hitmap[ipix] += n * iray

    return hitmap


# --- MONTE CARLO - IN PARTS ---
def create_random_source_map(source_bias, z, source_profile, rng):
    sources = np.zeros(source_bias.shape, dtype=int)
    npix = len(source_bias[0])
    nside = hp.npix2nside(npix)
    sangle = hp.nside2pixarea(nside)
    s0, sind = source_profile

    z = z.reshape((len(z), 1)) # Converting to proper column vector for ease of calculations
    dz = np.gradient(z, axis=0)
    
    dprop = z2dprop(z)
    dl = dprop * (1+z)
    dv = sangle * dl ** 2 / (1+z) * dz * C / H0 / sqrtH(z) / (1+z) # TODO dl->da?

    mean_source = source_bias * dv * s0 * (1+z) ** sind

    sources = rng.poisson(mean_source)

    # plt.figure()
    # plt.loglog(dprop, (np.sum(sources, axis=1)))

    return sources

def generate_rays_from_sources(sources, dndr, zs, exposure, source_profile, rng, smear=0):
    npix = len(sources[0])
    nside = hp.npix2nside(npix)
    s0, _ = source_profile

    # The exposure is in km^2, so we need to divide by this
    mpc_in_km = 3.086e19
    at = exposure / (mpc_in_km)**2 

    z = zs.reshape((len(zs), 1)) # Converting to proper column vector for ease of calculations
    dndr = dndr.reshape((len(z), 1))

    dprop = z2dprop(z)
    dl = dprop * (1+z)

    persource_flux = dndr / s0 * (1+z) / (4 * np.pi * dl ** 2) # TODO calculate this once and for all...
    full_flux = persource_flux * sources

    # dist_flux = np.sum(full_flux, axis=1)
    # plt.plot(dl, dist_flux)

    flux = np.sum(full_flux, axis=0)

    if smear != 0:
        # That 1e8 thing is to avoid overflow 
        flux = hp.smoothing(flux / 1e8, smear * np.pi/180) * 1e8
        flux = np.maximum(flux, 0)

    means = flux * at
    # print(means[0:3])
    # means += 0.5
    return rng.poisson(means)


# --- TEMP ---
def save_hitmap(hitmap, source_model, source_profile, idx, name, path):
    gind = source_model
    s0 = source_profile[0]
    logs0 = int(np.log10(s0))

    full_name = f"cr_{name}_m{gind}_s{logs0}_{idx}"
    full_path = path + "/" + full_name

    # Absolutely no way we get more than 10000 rays in one pixel
    hitmapint = hitmap.astype(np.uint16)
    np.save(full_path, hitmapint)


def strong_attempt(b, zs, at, source_model, source_profile, rng, save_path):
    nside = hp.get_nside(at)

    es = np.linspace(2e19, 8e19, 7)
    es[-1] = 2e22

    # FUCK U

    dndrs = []
    for i in range(len(es) - 1):
        dndrs.append(propagation.calc_cosmic_ray_rate_density(es[i], es[i+1], zs, source_model, 0, FUCKYOU=True))

    dndr_total = np.sum(dndrs, axis=0)
    dndr_total_h = np.sum(dndrs[2:], axis=0)

    dndrs_pro = []
    for i in range(len(es) - 1):
        dndrs_pro.append(propagation.calc_cosmic_ray_rate_density(es[i], es[i+1], zs, 0, 0, FUCKYOU=True))

    dndr_pro_total = np.sum(dndrs_pro, axis=0)
    dndr_pro_total_h = np.sum(dndrs_pro[2:], axis=0)

    lowe_dndr = [0, 1]
    highe_dndr = [2, 3, 4, 5]

    # plt.figure()
    # ax = plt.axes()
    # ax.set_xscale('log')
    # cumsums = [np.cumsum(dndr * np.gradient(zs)) / sum(dndr * np.gradient(zs)) for dndr in dndrs]
    # for i, cumsum in enumerate(cumsums):
    #     plt.plot(z2dprop(zs), cumsum, color='black', alpha=(i+1) / len(cumsums), label=f'{es[i]}')
    #     plt.hlines(0.9, 0, 1000, color='red')
    #     plt.show()
    #     plt.figure()
    #     ax = plt.axes()
    #     ax.set_xscale('log')
    
    # plt.legend()
    # plt.figure()
    # for i, dndr in enumerate(dndrs):
    #     plt.loglog(z2dprop(zs), dndr / sum(dndr * np.gradient(zs)), color='black', alpha=(i+1) / len(dndrs), label=f'{es[i]}')
    # plt.legend()

    # plt.figure()
    # ax = plt.axes()
    # ax.set_xscale('log')
    # cumsums = [np.cumsum(dndr * np.gradient(zs)) / sum(dndr * np.gradient(zs)) for dndr in dndrsp]
    # for i, cumsum in enumerate(cumsums):
    #     plt.plot(z2dprop(zs), cumsum, color='black', alpha=(i+1) / len(cumsums), label=f'{es[i]}')
    
    # plt.hlines(0.9, 0, 1000, color='red')
    # plt.legend()
    # plt.figure()
    # for i, dndr in enumerate(dndrsp):
    #     plt.loglog(z2dprop(zs), dndr / sum(dndr * np.gradient(zs)), color='black', alpha=(i+1) / len(dndrsp), label=f'{es[i]}')
    # plt.legend()

    # am = create_average_map(b, sum(dndrs), zs, at, source_profile)
    # print(sum(am))
    # hp.mollview(am)
    # hp.mollview(am / at)
    # at2 = at.copy()
    # for ipix in range(len(at2)):
    #     lon, lat = hp.pix2ang(nside, ipix, lonlat=True)
    #     if (np.abs(lat) < 5) or ((np.abs(lat) < 10 and np.abs((lon + 180) % 360 - 180) < 30)):
    #         at2[ipix] *= 0.3
    # hp.mollview(create_average_map(b, sum(dndrs), zs, at2, source_profile))
    

    # plt.show()
    # return

    lv2 = analysis.LocalVarianceTest2(gmf.deflection_random(4e18) * 180 / np.pi, hp.get_nside(at))
    # lv22 = analysis.LocalVarianceTest2(20, hp.get_nside(at))
    # lv222 = analysis.LocalVarianceTest2(30, hp.get_nside(at))
    # lv2222 = analysis.LocalVarianceTest2(40, hp.get_nside(at))
    lv2_results = []

    # Average "flux" map because the test doesn't actually care about the exposure, it's nice like that
    avg = create_average_flux_map(b, sum(dndrs), zs, source_profile)
    avg_pro = create_average_flux_map(b, sum(dndrs_pro), zs, source_profile)
    iso_b = np.outer(np.mean(b, axis=1), np.ones(len(avg)))
    iso_avg = create_average_flux_map(iso_b, sum(dndrs), zs, source_profile)

    mft_pro = analysis.BigMatchedFilterTest(avg_pro)
    mft_nuc = analysis.BigMatchedFilterTest(avg)
    mft_results = []
    hp.mollview(avg)
    hp.mollview(avg_pro)
    hp.mollview(iso_avg)
    mft_nuc.save("/home/nimrod/physics/uhecr/mf/nuc")
    mft_pro.save("/home/nimrod/physics/uhecr/mf/pro")
    plt.show()

    mtp = analysis.MultipolesTest()
    mtp_results = []

    totalmaps = []

    closest_source = []
    highest_hit = []

    closest_boys = []

    for i in tqdm.tqdm(range(1000)):
        sources = create_random_source_map(b, zs, source_profile, rng)
        # sources_iso = create_random_source_map(iso_b, zs, source_profile, rng)


        totalmap = generate_rays_from_sources(sources, dndr_total, zs, at, source_profile, rng, 0)
        totalmap_pro = generate_rays_from_sources(sources, dndr_pro_total, zs, at, source_profile, rng, 0)
        # totalmap_iso = generate_rays_from_sources(sources_iso, dndr_total, zs, at, source_profile, rng, 0)
        # plt.show()
        # totalmap_iso_pro = generate_rays_from_sources(sources_iso, dndr_pro_total, zs, at, source_profile, rng, 0)
        # hp.mollview(totalmap_iso)
        # hp.mollview(totalmap_iso_pro)
        # hp.mollview(avg_pro)
        # plt.show()

        # for i in []:
        for j, dndr in enumerate(dndrs):
            break
            hitmap = generate_rays_from_sources(sources, dndr, zs, at, source_profile, rng, 0)
            totalmap += hitmap
            # hitmap_pro = generate_rays_from_sources(sources, dndrs_pro[j], zs, at, source_profile, rng, 0)
            # totalmap_pro += hitmap_pro
            # hitmap_iso = generate_rays_from_sources(sources_iso, dndr, zs, at, source_profile, rng, 0)
            # hitmap_iso_pro = generate_rays_from_sources(sources_iso, dndrs_pro[j], zs, at, source_profile, rng, 0)
            
            if None != save_path:
                save_hitmap(hitmap, -2, source_profile, i, f"auger10_e{int(es[j] / 1e19)}", save_path)
                save_hitmap(hitmap_pro, 0, source_profile, i, f"auger10_e{int(es[j] / 1e19)}", save_path)
                save_hitmap(hitmap_iso, -2, source_profile, i, f"auger10I_e{int(es[j] / 1e19)}", save_path)
                save_hitmap(hitmap_iso_pro, 0, source_profile, i, f"auger10I_e{int(es[j] / 1e19)}", save_path)

        # lv2_results.append(lv2.test(totalmap))
        # mtp_results.append((mtp.test(totalmap), mtp.test(totalmap_pro), mtp.test(totalmap_iso)))
        # totalmap = generate_rays_from_sources(sources, sum(dndrs), zs, at, source_profile, rng)
        # lv2_results.append((
            # lv2.test(totalmap),
            # lv2.test(totalmap_pro),
            # lv2.test(totalmap_iso),
            # lv2.test(totalmap_iso_pro)
            # lv22.test(totalmap),
            # lv222.test(totalmap),
            # lv2222.test(totalmap),
        # ))
        # mat = mt.test(totalmap)
        # hp.mollview(mat)
        # hp.mollview(totalmap)
        # hp.mollview(avg)
        # plt.show()
        mft_results.append(np.array([
        #     mft_pro.test(totalmap_iso),
        #     mft_pro.test(totalmap_iso_pro),
            mft_pro.test(totalmap),
            mft_pro.test(totalmap_pro),
        #     mft_nuc.test(totalmap_iso),
        #     mft_nuc.test(totalmap_iso_pro),
        #     mft_nuc.test(totalmap),
            # mft_nuc.test(totalmap_pro),
        ]))

        # closest_source.append(np.argmax(np.sum(sources, axis=1) > 0))
        # highest_hit.append(max(totalmap))

        # if (closest_source[-1] == 0):
            # closest_ipix = np.argmax(sources[0])
            # closest_boys.append(closest_ipix)

        # totalmaps.append(totalmap)
        # print(np.sum(sources))
        # hp.mollview(totalmap, title='r')
        # print(sum(totalmap))
        # plt.show()

    # return

    # lvnuc, lvpro, lvin, lvip = zip(*lv2_results)
    # mn = min(lvnuc + lvpro + lvin + lvip)
    # mx = max(lvnuc + lvpro + lvin + lvip)
    # plt.hist(lvnuc, alpha=0.6, bins=np.linspace(mn, mx))
    # plt.hist(lvpro, alpha=0.6, bins=np.linspace(mn, mx))
    # plt.hist(lvin, alpha=0.6, bins=np.linspace(mn, mx))
    # plt.hist(lvip, alpha=0.6, bins=np.linspace(mn, mx))
    # plt.show()
    # return

    # mtp_nuc, mtp_pro, mtp_iso = zip(*mtp_results)
    # corner.corner(np.array(mtp_nuc))
    # corner.corner(np.array(mtp_pro))
    # corner.corner(np.array(mtp_iso))
    # plt.show()
    # return
    # mtp_results = np.array(mtp_results)

    mft_results = np.array(mft_results)
    nuu, pro = zip(*mft_results)
    plt.hist(nuu, bins=np.linspace(min(nuu + pro), max(nuu + pro)), alpha=.6)
    plt.hist(pro, bins=np.linspace(min(nuu + pro), max(nuu + pro)), alpha=.6)
    # np.save("/home/nimrod/physics/uhecr/bigone", mft_results)

    # closest_boys = np.array(closest_boys)
    # closest_source = np.array(closest_source)
    # highest_hit = np.array(highest_hit)

    # l1, l2, l3, l4, l5, l6, l7, l8 = zip(*mft_results)
    # l1, l2, l3, l4, l5, l6, l7, l8 = np.array(l1), np.array(l2), np.array(l3), np.array(l4), np.array(l5), np.array(l6), np.array(l7), np.array(l8)
    # plt.figure()
    # plt.hist(l1, bins=np.linspace(min(l1), max(l3)), alpha=0.6)
    # plt.hist(l2, bins=np.linspace(min(l1), max(l3)), alpha=0.6)
    # plt.hist(l3, bins=np.linspace(min(l1), max(l3)), alpha=0.6)
    # plt.hist(l4, bins=np.linspace(min(l1), max(l3)), alpha=0.6)
    # plt.figure()
    # plt.hist(l5, bins=np.linspace(min(l5), max(l7)), alpha=0.6)
    # plt.hist(l6, bins=np.linspace(min(l5), max(l7)), alpha=0.6)
    # plt.hist(l7, bins=np.linspace(min(l5), max(l7)), alpha=0.6)
    # plt.hist(l8, bins=np.linspace(min(l5), max(l7)), alpha=0.6)

    # plt.figure()
    # plt.hist(l5 - l1, bins=np.linspace(min(l5 - l1), max(l7 - l3)), alpha=0.6)
    # plt.hist(l6 - l2, bins=np.linspace(min(l5 - l1), max(l7 - l3)), alpha=0.6)
    # plt.hist(l7 - l3, bins=np.linspace(min(l5 - l1), max(l7 - l3)), alpha=0.6)
    # plt.hist(l8 - l4, bins=np.linspace(min(l5 - l1), max(l7 - l3)), alpha=0.6)

    # print(f"SNR pro vs iso {(np.mean(l4) - np.mean(l2))/np.std(l2)}")
    # print(f"SNR nuc vs pro {(np.mean(l7 - l3) - np.mean(l8 - l4))/np.std(l8 - l4)}")
    
    # print(np.mean(l7 - l3))
    # print(np.mean(l8 - l4))
    # print(np.std(l7 - l3))
    # print(np.std(l8 - l4))

    plt.show()
    # return

    lv2_results = np.array(lv2_results)
    # plt.figure()
    plt.figure()
    plt.hist(lv2_results, bins=np.linspace(min(lv2_results), 3))#max(lv2_results)))
    # np.save("lv2_s", lv2_results)
    
    thresh = 0.0045
    # print(f"low energy overd={thresh} pvalue: {pvalue}")

    plt.show()
    return

    totalmaps = np.array(totalmaps)
    stds = np.std(totalmaps, axis=0)
    means = np.mean(totalmaps, axis=0)
    hp.mollview(means / stds)
    #hp.mollview(means / (stds * stds))
    avv = create_average_map(b, np.sum(dndrs, axis=0), zs, at, source_profile)
    
    hp.mollview(np.sqrt(avv))
    hp.mollview(stds)

    hp.mollview(np.sqrt(at))
    plt.show()

    fig, sp = plt.subplots(2, 3)
    for i in range(len(es) - 1):
        f_overcounts = overcounts[-i-1]
        f_undercounts = undercounts[-i-1]
        rats = f_overcounts / f_undercounts / tp[0]
        iso_pvalue = np.sum(np.array(rats) < 1) / len(rats)
        
        sp.ravel()[i].set_title(f"{es[i]}, p={iso_pvalue}")
        sp.ravel()[i].hist(rats, bins=np.linspace(0, 5, num=50))
        
    fig, sp = plt.subplots(2, 3)
    for i, e in enumerate(es[:-1]):
        sp.ravel()[i].set_title(f"{e}")
        sp.ravel()[i].hist(diffs.transpose()[i])

    plt.show()


# --- MAIN ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-path", "-c", help="path for catalog", default="/home/nimrod/physics/uhecr/biteau/table5.dat")
    parser.add_argument("--nside", "-n", help="nside for the healpix map", default=NSIDE)
    parser.add_argument("--source-model", "-s", help="model name for source emission", type=int, default=SOURCE_MODEL)
    parser.add_argument("--exposure", "-e", choices=['isotropic', 'auger', 'auger10', 'ta', '2022'], help="sky exposure pattern to use (default: isotropic)", default='isotropic')
    parser.add_argument("--source-density", "-sd", help="log10 source density (Mpc^-3)", type=float, default=-2.0)
    parser.add_argument("--source-evolution", "-se", help="source evolution index", type=float, default=0.0)
    parser.add_argument("--output-directory", "-o", help="output path to save results in", default=None)
    return parser.parse_args()

def main():

    args = parse_args()
    rng = np.random.default_rng()

    zs = np.linspace(0, 0.4, 401)[1:] # Important: resolution need to be better than the bias map voxel size

    # 200 seems more or less the limit where the avg angular separation under 10 deg
    b = lss.create_source_bias_map_mrs("MRS/catalog/2mrs_1175_done.dat", "MRS/CORRECTIONS/nearby.txt", args.nside, zs, 0.5, 200)
    bl = lss.create_source_bias_map_mrsl("MRS/catalog/2mrs_1175_done.dat", "MRS/CORRECTIONS/nearby.txt", args.nside, zs, 0.5, 200)
    # b2 = lss.create_source_bias_map(args.catalog_path, args.nside, zs, 0.5, 300)
    
    # print(plot_average_map(args.nside, b, zs, 1.6e19, 3.2e19, -2, (1e-4, 0), "isotropic"))
    print(plot_average_map(args.nside, bl, zs, 1.6e19, 3.2e19, -2, (1e-4, 0), "isotropic"))
    # print(plot_average_map(args.nside, b, zs, 3.2e19, 2e21, -2, (1e-4, 0), "isotropic"))
    print(plot_average_map(args.nside, bl, zs, 3.2e19, 2e21, -2, (1e-4, 0), "isotropic"))

    # print(plot_average_map(args.nside, b, zs, 1.6e19, 3.2e19, 0, (1e-4, 0), "isotropic"))
    print(plot_average_map(args.nside, bl, zs, 1.6e19, 3.2e19, 0, (1e-4, 0), "isotropic"))
    # print(plot_average_map(args.nside, b, zs, 3.2e19, 2e21, 0, (1e-4, 0), "isotropic"))
    print(plot_average_map(args.nside, bl, zs, 3.2e19, 2e21, 0, (1e-4, 0), "isotropic"))
    plt.show()
    # e0s = np.logspace(19.3, 20.1, 9)
    # ns = []
    # ns2 = []
    # for e0 in e0s:
    #     ns.append(plot_average_map(args.nside, b, zs, e0, 2e21, -2, (1e-4, 0), "auger10"))
    #     ns2.append(plot_average_map(args.nside, bl, zs, e0, 2e21, -2, (1e-4, 0), "auger10"))
    # print(e0s)
    # print(ns)
    # print(ns2)
    # return
    # plt.xscale('log')
    # print(plot_average_map(args.nside, b, zs, 4e19, 2e21, -2, (1e-2, 0), "isotropic"))
    # plot_average_map(args.nside, b, zs, 8e19, 2e21, -2, (1e-2, 0), "isotropic")
    # print(plot_average_map(args.nside, b2, zs, 4e19, 2e21, -2, (1e-2, 0), "isotropic"))
    # plot_average_map(args.nside, b2, zs, 8e19, 2e21, -2, (1e-2, 0), "isotropic")

    # hp.mollview(b[10],title='b')
    # hp.mollview(b2[10],title='b2')
    # plt.show()
    # return
    # plot_average_map(args.nside, b, zs, 3.2e19, 2e21, -2, "isotropic")
    # plot_average_map(args.nside, b2, zs, 2.8e19, 2e21, -2, "auger")

    # plt.show()
    # return
    
    source_profile = (np.power(10, args.source_density), args.source_evolution)

    source_model = args.source_model
    at = exposure.create_exposure_map(args.nside, args.exposure)
    strong_attempt(bl, zs, at, source_model, source_profile, rng, args.output_directory)
    return

if __name__ == "__main__":
    main()