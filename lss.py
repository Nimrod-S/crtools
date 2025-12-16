import argparse

import numpy as np
import scipy as sp
import healpy as hp

from cosmology import *

from matplotlib import pyplot as plt


# --- LOCAL STRUCTURE ---
def selection_func_2mrs(z):
    A, alpha, gamma, zc = 116000, 2.108, 1.125, 0.025 # TODO citation
    return selection_func_generic(z, A, alpha, gamma, zc)

def selection_func_fit(z):
    A, alpha, gamma, zc = 1.72511352e+06, 1.63834331e+00, 1.67935457e+00, 2.38702796e-02
    return selection_func_generic(z, A, alpha, gamma, zc)

def selection_func_generic(z, A, alpha, gamma, zc):
    return A * np.power(z, gamma) * np.exp(-np.power(z / zc, alpha))

def lum_selection_func_fit(dl):
    A, alpha, dlc = .05074186, 2.46522663, 110.77033631159263
    return lum_selection_func_generic(dl, A, alpha, dlc)

def lum_selection_func_generic(dl, A, alpha, dlc):
    return A * sp.special.gammaincc(alpha, (dl / dlc)**2)

# Numbers from "Reconstructed density and velocity fields from the 2MASS Redshift Survey"
V_LG = 627
L_LG = 273
B_LG = 29
VEC_LG = hp.ang2vec(L_LG, B_LG, lonlat=True)
def fix_localgroup_v(l, b, v):
    vecgal = hp.ang2vec(l, b, lonlat=True)
    cos = np.dot(VEC_LG, vecgal)
    return v + cos * V_LG

def parse_catalog_data_mrs_pure(catalog_path):
    data = []
    with open(catalog_path) as f:
        for entry in f:
            if entry[0] == "#":
                continue
            fields = entry.split()

            data.append({
                "name": fields[-1],
                "l": float(fields[3]),  # Galactic longitude (degrees)
                "b": float(fields[4]),  # Galactic latitude (degrees)
                "v": float(fields[24]), # Redshift
                "Ks": float(fields[5])
            })

    return data

def parse_catalog_data_mrs(catalog_path, correction_path):
    corrections = {}
    with open(correction_path) as f:
        for line in f:
            fields = line.split()
            corrections[fields[0]] = float(fields[1])

    data = []
    with open(catalog_path) as f:
        for entry in f:
            if entry[0] == "#":
                continue
            fields = entry.split()

            if fields[0] in corrections:
                modc = corrections[fields[0]]
                z = np.power(10, modc / 5 - 5) * H0 / C
                # TODO extra bias to fix here
            else:
                real_v = fix_localgroup_v(float(fields[3]), float(fields[4]), float(fields[24]))
                z = max(real_v / C, 0)

            # if float(fields[5]) > 11.25:
            #     continue

            data.append({
                "name": fields[-1],
                "Ks": float(fields[5]),
                "l": float(fields[3]),  # Galactic longitude (degrees)
                "b": float(fields[4]),  # Galactic latitude (degrees)
                "z": z                  # Redshift
            })

    return data

def fill_galactic_plane(catalog_data, min_z, max_z):
    # Based on: Crook et al. 2007 (arXiv:astro-ph/0610732)
    # The longitude of a bin represents the smaller edge
    lon_bins = np.linspace(0, 360, 37)
    lat_internal = np.array(3*[10] + 30*[5] + 3*[10])
    lat_external = np.array(3*[20] + 30*[15] + 3*[20])
    mean_factor = np.array(3*[1] + 30*[0.5] + 3*[1])
    z_bins = np.linspace(min_z, max_z, int((max_z - min_z) / 0.0033)) # Rough number choice
    counts_internal = np.zeros((len(z_bins) - 1, 36)) # The ones we already see inside the plane
    counts_external = np.zeros((len(z_bins) - 1, 36))
    magnitudes_external = [[[] for _ in range(36)] for _ in range(len(z_bins) - 1)]

    for entry in catalog_data:
        z = entry['z']
        if z > max_z or z < min_z:
            continue
        zbin = np.argmin(z > z_bins) - 1
        
        l, b = entry['l'], entry['b']
        lbin = np.argmin(l > lon_bins) - 1

        if np.abs(b) > lat_external[lbin]:
            continue
        if np.abs(b) > lat_internal[lbin]:
            counts_external[zbin][lbin] += 1
            magnitudes_external[zbin][lbin].append(entry['Ks'])
        else:
            counts_internal[zbin][lbin] += 1

    # Cosine is a geometrical factor since the upper latitudes are effectively a little smaller
    missing_counts = counts_external * mean_factor / np.cos(np.pi/180 * lat_internal) - counts_internal
    missing_N = np.where(counts_external != 0,
        sp.stats.norm(loc=missing_counts, scale=1).rvs(),
        0
    ) # TODO CHANGE

    for li in range(len(lon_bins) - 1):
        for zi in range(len(z_bins) - 1):
            N = int(missing_N[zi][li] + 0.5)
            if N <= 0:
                continue
            ls = sp.stats.uniform(loc=lon_bins[li], scale=lon_bins[li + 1]-lon_bins[li]).rvs(size=N)
            bs = sp.stats.uniform(loc=-lat_internal[li], scale=lat_internal[li]*2).rvs(size=N)
            zs = sp.stats.uniform(loc=z_bins[zi], scale=z_bins[zi + 1]-z_bins[zi]).rvs(size=N)
            ks = np.random.choice(magnitudes_external[zi][li], N) # replace=True. TODO CHANGE
            for l, b, z, k in zip(ls, bs, zs, ks):
                catalog_data.append({
                    "name": "FAKE",
                    "l": l,  # Galactic longitude (degrees)
                    "b": b,  # Galactic latitude (degrees)
                    "z": z,  # Redshift
                    "Ks": k     # K band magnitude
                })


def create_source_bias_map_mrsl(catalog_path, correction_path, nside, zs, min_catalog_dl, max_catalog_dl, fill_plane=True, bias_ratio=1):
    """
    Below min_catalog_dl, assume there is nothing (basically cut out our satellite dwarf galaxies)
    Above max_catalog_dl, assume isotropy
    """
    data = parse_catalog_data_mrs(catalog_path, correction_path)

    dls = z2dprop(zs) * (1+zs)

    npix = hp.nside2npix(nside)
    sangle = hp.nside2pixarea(nside)
    hpmaps = np.zeros((len(zs), npix))
    galcounts = np.zeros(len(zs))
    totallum = 0

    if (fill_plane):
        fill_galactic_plane(data, min_catalog_dl * H0 / C, max_catalog_dl * H0 / C)

    # Build a healpy map
    for entry in data:
        z = entry['z']
        dl = z2dprop(z) * (1 + z)
        if dl > max_catalog_dl or dl < min_catalog_dl:
            continue
        if dl > dls[-1]:
            print("PROBLEM")
            continue
        iz = np.argmax(dl <= dls)
        l, b = entry['l'], entry['b'] # deg
        ipix = hp.ang2pix(nside, l, b, lonlat=True)
        
        k = -6 * np.log(1 + z) # k-correction
        l = np.power(10, -0.4 * (entry['Ks'] - k)) * dl**2 # The "actual" luminosity in useful units (erg/s for example) is of course some factor times this but we are going to normalize it anyway.
        l /= lum_selection_func_fit(dl)
        hpmaps[iz][ipix] += l
        totallum += l

        galcounts[iz] += 1 

        
    # Right now, hpmaps is not volume-normalized and does not represent density, just total luminosity. We will deal with this later during normalization.
    normalized = False
    for i, dl in enumerate(dls):
        if dl <= max_catalog_dl and galcounts[i] != 0:
            # Smooth:
            lmean = (4 * np.pi * dl ** 2 / galcounts[i]) ** (1/2) # Roughly mean distance, in mpc (suspicious geometric factor)
            lmean = max(lmean, 10)

            hpmaps[i] = hp.sphtfunc.smoothing(hpmaps[i], sigma=lmean / dl)

            # Rarely, numerical artifacts of the smoothing produces some negative values. From experience they are all <1e-4 of the mean so this is probably not that bad. We HAVE to fix it or the poisson distributions will get messed up later. TODO
            hpmaps[i] = np.maximum(hpmaps[i], 0)

            hpmaps[i] /= dl ** 2 * sangle
            
        if dl > max_catalog_dl:
            # We are out, now normalize and start being isotropic:
            if not normalized:
                normalized = True
                mean_density = totallum / np.sum(4 * np.pi * dls[:i] ** 2) # I thought about this a lot, this is correct
                hpmaps[:i] /= mean_density

                # hpmaps[:i] = (hpmaps[:i] / mean_density - 1) * bias_ratio + 1
                # hpmaps[:i][hpmaps[:i] < 0] = 0
        
            # Just fill with isotropic value
            hpmaps[i] = np.ones(npix)
            continue

    hpmaps = apply_linear_bias(hpmaps, bias_ratio)
    return hpmaps

def apply_linear_bias(m, bias_ratio):
    b = m.copy()
    b = (b - 1) * bias_ratio + 1
    b[b < 0] = 0
    return b


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nside", "-n", help="nside for the healpix map", default=32)
    parser.add_argument("--output-directory", "-o", help="output path to save results in", default="/home/nimrod/physics/uhecr/cr_output")
    parser.add_argument("--mrs-directory", "-m", help="path for MRS catalog + correction files", default="/home/nimrod/physics/uhecr/MRS")
    args = parser.parse_args()
    
    zs = np.linspace(0, 0.4, 401)[1:] # Important: resolution need to be better than the bias map voxel size

    # 200 seems more or less the limit where the avg angular separation under 10 deg
    b = create_source_bias_map_mrsl(args.mrs_directory+"/catalog/2mrs_1175_done.dat", args.mrs_directory+"/CORRECTIONS/nearby.txt",
                                     args.nside, zs, 0.5, 200)
    
    np.save(args.output_directory+"/lss/lss_bias_v1", b)


if __name__ == "__main__":
    main()
