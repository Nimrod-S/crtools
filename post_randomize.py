import math
import argparse
import os
import tqdm
import numpy as np
import scipy as sp
from scipy.special import exp1
from matplotlib import pyplot as plt
import healpy as hp

import analysis
import exposure
import propagation
import gmf
from cosmology import *

NSIDE=32
SOURCE_MODEL=-2


# --- DATA ---
def load_hitmap(source_model, exposure, source_profile, name, root_path):
    path = root_path + f"/hitmaps/{exposure}"

    gind = source_model
    s0 = source_profile[0]
    logs0 = int(np.log10(s0))
    bratio = source_profile[2]

    pattern = f"cr_{name}_m{gind}_s{logs0}_b{bratio}_"
    fnames = [fname for fname in os.listdir(path) if pattern in fname]
    
    fnames.sort(key=lambda fname:int(fname.split('_')[-1][:-4])) # sort by idx

    for fname in fnames:
        full_path = path + "/" + fname
        yield np.load(full_path).astype(np.int64) # We are going to start summing and averaging stuff, so back to serious numbers lol
    return

def load_meanmap(source_model, exposure, source_profile, name, root_path):
    path = root_path + f"/meanmaps/{exposure}"

    gind = source_model
    s0 = source_profile[0]
    logs0 = int(np.log10(s0))
    bratio = source_profile[2]

    name = f"mean_{name}_m{gind}_s{logs0}_b{bratio}.npy"

    full_path = path + "/" + name
    return np.load(full_path)

def save_result(data, exposure, source_model, source_profile, testname, root_path):
    path = root_path + f"/results/{exposure}"
    gind = source_model
    s0 = source_profile[0]
    logs0 = int(np.log10(s0))
    bratio = source_profile[2]


    full_name = f"{testname}_b{bratio}_m{gind}_s{logs0}"
    full_path = path + "/" + full_name

    np.save(full_path, data)

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
    parser.add_argument("--exposure", "-e", choices=['isotropic', 'auger', 'auger10', 'ta', '2022', 'auger2', 'ideal'], help="sky exposure pattern to use (default: isotropic)", default='isotropic')
    parser.add_argument("--source-density", "-sd", help="log10 source density (Mpc^-3)", type=float, default=-2.0)
    parser.add_argument("--source-evolution", "-se", help="source evolution index", type=float, default=0.0)
    parser.add_argument("--bias", "-b", choices=['iso', 'neutral', 'high'], help="source distribution bias to 2MRS", default='neutral')
    parser.add_argument("--input-directory", "-i", help="path with randomize_rays.py results", default="./cr_output")
    
    parser.add_argument("--gmf", "-g", type=int)
    parser.add_argument("--ecal", "-c", type=int, default=2)

    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--ect", action="store_true")
    parser.add_argument("--mpt", action="store_true")
    parser.add_argument("--mftc", action="store_true")
    parser.add_argument("--ent", action="store_true")
    return parser.parse_args()

def main():

    args = parse_args()

    at = exposure.create_exposure_map(args.nside, args.exposure)
    bratio = {"iso": 0, "neutral": 1, "high": 1.7}[args.bias]

    source_profile = (np.power(10, args.source_density), args.source_evolution, bratio)
    es = np.load(args.input_directory+"/flux/energies_v2.npy")

    magname = "" if (args.gmf == -1) else f"_U{args.gmf}"
    mask = 10 if args.mask else 0

    iters_nuc = iter(load_hitmap(args.source_model, args.exposure, source_profile, f"v2"+magname, args.input_directory))
    iters_pro = iter(load_hitmap(0, args.exposure, source_profile, f"v2"+magname, args.input_directory))

    MFTC = args.mftc
    ECT = args.ect
    MPT = args.mpt
    ENT = args.ent

    if MFTC:
        compare_n = load_meanmap(-2, args.exposure, (1e-2, 0, 1), f"v2", args.input_directory)
        compare_p = load_meanmap(0, args.exposure, (1e-2, 0, 1), f"v2", args.input_directory)
        mfnuc = analysis.BigMatchedFilterTest(mask, np.sum(compare_n[args.ecal:], axis=0))
        mfpro = analysis.BigMatchedFilterTest(mask, np.sum(compare_p[args.ecal:], axis=0))
        
        nuc_vs_nuc = []
        nuc_vs_pro = []
        pro_vs_nuc = []
        pro_vs_pro = []
    if ECT:
        ect_angle = 16.7
        ect = analysis.SmallCorrelationTest(ect_angle, mask, args.nside)

        ect_nuc = []
        ect_pro = []
    if MPT:
        mpt_angle = 16.7
        mpt = analysis.MultipolesTest(mpt_angle, at)
        
        mpt_nuc = []
        mpt_pro = []
        dst_nuc = []
        dst_pro = []
    if ENT:
        ent_angle = 16.7
        ent = analysis.SmallScaleVarTest(ent_angle, args.nside, mask)

        ent_nuc = []
        ent_pro = []
        ent_nuc_s = []
        ent_pro_s = []


    #for hitmaps in tqdm.tqdm(zip(*iters), total=10000):
    for hitmaps in iters_nuc:
        if MFTC:
            midmap = np.sum(hitmaps[args.ecal:], axis=0)
            nuc_vs_nuc.append(mfnuc.test(midmap))
            nuc_vs_pro.append(mfpro.test(midmap))
        if ECT:
            lowmap = np.sum(hitmaps[args.ecal:args.ecal+2], axis=0)
            highmap = np.sum(hitmaps[args.ecal+2:], axis=0)
            ect_nuc.append(ect.test_against(lowmap, highmap))
        if MPT:
            midmap = np.sum(hitmaps[args.ecal:], axis=0)
            lowmap = np.sum(hitmaps[0:4], axis=0)

            dip = mpt.dipole_semiexp(midmap)
            mpt_nuc.append(dip)
            dipl = mpt.dipole_semiexp(lowmap)
            dst_nuc.append(dipl)
        if ENT:
            midmap = np.sum(hitmaps[args.ecal:], axis=0)
            x, s = ent.test_ent(midmap)
            ent_nuc.append(x)
            ent_nuc_s.append(s)


    #for hitmaps in tqdm.tqdm(zip(*iters_pro), total=10000):
    for hitmaps in iters_pro:

        if MFTC:
            midmap = np.sum(hitmaps[args.ecal:], axis=0)
            pro_vs_nuc.append(mfnuc.test(midmap))
            pro_vs_pro.append(mfpro.test(midmap))
        if ECT:
            lowmap = np.sum(hitmaps[args.ecal:args.ecal+2], axis=0)
            highmap = np.sum(hitmaps[args.ecal+2:], axis=0)
            ect_pro.append(ect.test_against(lowmap, highmap))
        if MPT:
            midmap = np.sum(hitmaps[args.ecal:], axis=0)
            lowmap = np.sum(hitmaps[0:4], axis=0)

            dip = mpt.dipole_semiexp(midmap)
            mpt_pro.append(dip)
            dipl = mpt.dipole_semiexp(lowmap)
            dst_pro.append(dipl)
        if ENT:
            midmap = np.sum(hitmaps[args.ecal:], axis=0)
            x, s = ent.test_ent(midmap)
            ent_pro.append(x)
            ent_pro_s.append(s)


    if MFTC:
        nuc_vs_nuc = np.array(nuc_vs_nuc)
        nuc_vs_pro = np.array(nuc_vs_pro)
        pro_vs_nuc = np.array(pro_vs_nuc)
        pro_vs_pro = np.array(pro_vs_pro)

        txt = {2: "mid", 1: "low", 3: "high"}[args.ecal]
        save_result(nuc_vs_nuc, args.exposure, args.source_model, source_profile, f"mfsnuc_k{mask}_{txt}"+magname, args.input_directory)
        save_result(pro_vs_nuc, args.exposure, 0, source_profile, f"mfsnuc_k{mask}_{txt}"+magname, args.input_directory)

        save_result(nuc_vs_pro, args.exposure, args.source_model, source_profile, f"mfspro_k{mask}_{txt}"+magname, args.input_directory)
        save_result(pro_vs_pro, args.exposure, 0, source_profile, f"mfspro_k{mask}_{txt}"+magname, args.input_directory)

    if ECT:
        ect_nuc = np.array(ect_nuc)
        ect_pro = np.array(ect_pro)

        txt = {2: "mid", 1: "low", 3: "high"}[args.ecal]

        save_result(ect_nuc, args.exposure, args.source_model, source_profile, f"ec{ect_angle}_k{mask}_{txt}"+magname, args.input_directory)
        save_result(ect_pro, args.exposure, 0, source_profile, f"ec{ect_angle}_k{mask}_{txt}"+magname, args.input_directory)

    if MPT:
        txt = {2: "mid", 1: "low", 3: "high", 0: "all"}[args.ecal]

        mpt_nuc = np.array(mpt_nuc)
        mpt_pro = np.array(mpt_pro)

        save_result(mpt_nuc, args.exposure, args.source_model, source_profile, f"mpd{mpt_angle}_{txt}"+magname, args.input_directory)
        save_result(mpt_pro, args.exposure, 0, source_profile, f"mpd{mpt_angle}_{txt}"+magname, args.input_directory)

        dst_nuc = np.array(dst_nuc)
        dst_pro = np.array(dst_pro)

        save_result(dst_nuc, args.exposure, args.source_model, source_profile, f"mpd{mpt_angle}_e2e42"+magname, args.input_directory)
        save_result(dst_pro, args.exposure, 0, source_profile, f"mpd{mpt_angle}_e2e42"+magname, args.input_directory)

    if ENT:
        ent_nuc = np.array(ent_nuc)
        ent_pro = np.array(ent_pro)

        txt = {2: "mid", 1: "low", 3: "high", 0: "all"}[args.ecal]

        save_result(ent_nuc, args.exposure, -2, source_profile, f"en3{ent_angle}_k{mask}_{txt}"+magname, args.input_directory)
        save_result(ent_pro, args.exposure, 0, source_profile, f"en3{ent_angle}_k{mask}_{txt}"+magname, args.input_directory)
        save_result(ent_nuc_s, args.exposure, -2, source_profile, f"ens3{ent_angle}_k{mask}_{txt}"+magname, args.input_directory)
        save_result(ent_pro_s, args.exposure, 0, source_profile, f"ens3{ent_angle}_k{mask}_{txt}"+magname, args.input_directory)

    

if __name__ == "__main__":
    main()
