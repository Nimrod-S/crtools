import math
import argparse
import os
import tqdm
import numpy as np
import scipy as sp
from scipy.special import exp1
from matplotlib import pyplot as plt
import healpy as hp
import corner

import analysis
import exposure
import propagation
import gmf
from cosmology import *

NSIDE=32
SOURCE_MODEL=-2


# --- DATA ---
def load_hitmap(source_model, source_profile, name, path):
    gind = source_model
    s0 = source_profile[0]
    logs0 = int(np.log10(s0))

    pattern = f"cr_{name}_m{gind}_s{logs0}_"
    fnames = [fname for fname in os.listdir(path) if pattern in fname]
    
    fnames.sort(key=lambda fname:int(fname.split('_')[-1][:-4])) # sort by idx

    for fname in fnames:
        full_path = path + "/" + fname
        yield np.load(full_path).astype(np.int64) # We are going to start summing and averaging stuff, so back to serious numbers lol
    return


# --- MISC ---
def fancy_smooth(hitmap, e0):
    r = np.logspace(1, 19.5)
    rdist = propagation.get_r_dist(e0, r, -2)
    rdist /= np.sum(rdist) # TODO maybe get_r_dist is already normalized
    dr = np.gradient(r)

    result = 0
    for i in range(len(r)):
        th = gmf.deflection_random(r[i])
        results += hp.smoothing(hitmap, sigma=th) * rdist[i] * dr[i]


# --- MAIN ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", "-s", help="model name for source emission", type=int, default=SOURCE_MODEL)
    parser.add_argument("--nside", "-n", help="nside for the healpix map", default=NSIDE)
    parser.add_argument("--exposure", "-e", choices=['isotropic', 'auger', 'auger10', 'ta', '2022'], help="sky exposure pattern to use (default: isotropic)", default='isotropic')
    parser.add_argument("--source-density", "-sd", help="log10 source density (Mpc^-3)", type=float, default=-2.0)
    parser.add_argument("--source-evolution", "-se", help="source evolution index", type=float, default=0.0)
    parser.add_argument("--input-directory", "-i", help="path with randomize_rays.py results", default="./cr_output")
    return parser.parse_args()

def main():

    args = parse_args()

    at = exposure.create_exposure_map(args.nside, args.exposure)

    hits = []
    
    source_profile = (np.power(10, args.source_density), args.source_evolution)
    es = np.linspace(2e19, 8e19, 7)
    es[-1] = 2e22
    iters = []
    iters_pro = []
    for e in es[:-1]:
        iters.append(iter(load_hitmap(args.source_model, source_profile, f"{args.exposure}_e{int(e / 1e19)}", args.input_directory)))
        iters_pro.append(iter(load_hitmap(0, source_profile, f"{args.exposure}_e{int(e / 1e19)}", args.input_directory)))

    vvv = np.zeros(hp.nside2npix(args.nside))

    mfnuc = analysis.BigMatchedFilterTest.load("/home/nimrod/physics/uhecr/mf/nuc")
    sc = analysis.SmallCorrelationTest()

    mt = analysis.MultipolesTest()

    res = []

    totalnuclow = np.zeros(len(at))
    totalnuchigh = np.zeros(len(at))
    totalprolow = np.zeros(len(at))
    totalprohigh = np.zeros(len(at))

    for hitmaps in tqdm.tqdm(zip(*iters), total=10000):
        low_e_hitmap = sum(hitmaps)
        high_e_hitmap = sum(hitmaps[2:])
        # res.append(sc.test_against(low_e_hitmap, high_e_hitmap))

        # res.append(mt.test(low_e_hitmap))
        # hp.mollview(low_e_hitmap)
        # hp.mollview(high_e_hitmap)
        # plt.show()
        totalnuclow += low_e_hitmap / np.sum(low_e_hitmap)
        totalnuchigh += high_e_hitmap / np.sum(high_e_hitmap)

    res2 = []

    for hitmaps in tqdm.tqdm(zip(*iters_pro), total=10000):
        low_e_hitmap = sum(hitmaps)
        high_e_hitmap = sum(hitmaps[2:])
        # res2.append(sc.test_against(low_e_hitmap, high_e_hitmap))

        # res2.append(mt.test(low_e_hitmap))
        # hp.mollview(low_e_hitmap)
        # hp.mollview(high_e_hitmap)
        # plt.show()
        totalprolow += low_e_hitmap / np.sum(low_e_hitmap)
        totalprohigh += high_e_hitmap / np.sum(high_e_hitmap)

    # corner.corner(np.array(res))
    # plt.show()
    # corner.corner(np.array(res2))
    # plt.show()


    hp.mollview(totalnuclow)
    hp.mollview(totalnuchigh)
    hp.mollview(totalprolow)
    hp.mollview(totalprohigh)
    plt.show()
    return

    res = np.array(res)
    res2 = np.array(res2)

    b = np.linspace(min(res2), max(res))
    plt.hist(res, alpha=0.6, bins=b)
    plt.hist(res2, alpha=0.6, bins=b)
    plt.show()
    

if __name__ == "__main__":
    main()