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
def load_hitmap(source_model, exposure, source_profile, name, root_path):
    path = root_path + "/maps"

    gind = source_model
    s0 = source_profile[0]
    logs0 = int(np.log10(s0))

    pattern = f"cr_{exposure}_{name}_m{gind}*_s{logs0}_"
    fnames = [fname for fname in os.listdir(path) if pattern in fname]
    
    fnames.sort(key=lambda fname:int(fname.split('_')[-1][:-4])) # sort by idx

    for fname in fnames:
        full_path = path + "/" + fname
        yield np.load(full_path).astype(np.int64) # We are going to start summing and averaging stuff, so back to serious numbers lol
    return

def save_result(data, exposure, source_model, source_profile, name, root_path):
    path = root_path + "/results"
    gind = source_model
    s0 = source_profile[0]
    logs0 = int(np.log10(s0))


    full_name = f"{name}_{exposure}_m{gind}*_s{logs0}"
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
    parser.add_argument("--exposure", "-e", choices=['isotropic', 'auger', 'auger10', 'ta', '2022'], help="sky exposure pattern to use (default: isotropic)", default='isotropic')
    parser.add_argument("--source-density", "-sd", help="log10 source density (Mpc^-3)", type=float, default=-2.0)
    parser.add_argument("--source-evolution", "-se", help="source evolution index", type=float, default=0.0)
    parser.add_argument("--input-directory", "-i", help="path with randomize_rays.py results", default="./cr_output")
    return parser.parse_args()

def main():

    args = parse_args()

    at = exposure.create_exposure_map(args.nside, args.exposure)

    source_profile = (np.power(10, args.source_density), args.source_evolution)
    es = np.linspace(2e19, 8e19, 7)
    es[-1] = 2e22
    iters = []
    iters_pro = []
    for e in es[:-1]:
        iters.append(iter(load_hitmap(args.source_model, args.exposure, source_profile, f"e{int(e / 1e19)}", args.input_directory)))
        iters_pro.append(iter(load_hitmap(0, args.exposure, source_profile, f"e{int(e / 1e19)}", args.input_directory)))


    MFT = True
    MFTI = True
    LVT = False
    MPT = False
    ECT = False


    if MFT:
        mfnuc = analysis.BigMatchedFilterTest.load(f"cr_output/patterns/mf_{args.exposure}_e6_nuc")
        mfpro = analysis.BigMatchedFilterTest.load(f"cr_output/patterns/mf_{args.exposure}_e6_pro")

        nuc_vs_nuc = []
        nuc_vs_pro = []
        pro_vs_nuc = []
        pro_vs_pro = []
    if MFTI: 
        mfiso = analysis.BigMatchedFilterTest.load(f"cr_output/patterns/mf_{args.exposure}_e6_iso")

        nuc_vs_iso = []
        pro_vs_iso = []
    if LVT:
        lvt_angle = 12
        lvt = analysis.LocalVarianceTest(lvt_angle, args.nside)

        lvt_nuc = []
        lvt_pro = []
    if MPT:
        mpt = analysis.MultipolesTest()

        mpt_nuc = []
        mpt_pro = []
    if ECT:
        ect_angle = 12
        ect = analysis.SmallCorrelationTest(ect_angle, args.nside)

        ect_nuc = []
        ect_pro = []


    for hitmaps in tqdm.tqdm(zip(*iters), total=10000):
        low_e_hitmap = sum(hitmaps[4:]) # TODO

        if MFT:
            nuc_vs_nuc.append(mfnuc.test(low_e_hitmap))
            nuc_vs_pro.append(mfpro.test(low_e_hitmap))
        if MFTI:
            nuc_vs_iso.append(mfiso.test(low_e_hitmap))
        if LVT:
            lvt_nuc.append(lvt.test(low_e_hitmap))
        if MPT:
            mpt_nuc.append(mpt.test(low_e_hitmap))
        if ECT:
            high_e_hitmap = sum(hitmaps[2:])
            ect_nuc.append(ect.test_against(low_e_hitmap, high_e_hitmap))


    for hitmaps in tqdm.tqdm(zip(*iters_pro), total=10000):
        low_e_hitmap = sum(hitmaps[4:]) # TODO

        if MFT:
            pro_vs_nuc.append(mfnuc.test(low_e_hitmap))
            pro_vs_pro.append(mfpro.test(low_e_hitmap))
        if MFTI:
            pro_vs_iso.append(mfiso.test(low_e_hitmap))
        if LVT:
            lvt_pro.append(lvt.test(low_e_hitmap))
        if MPT:
            mpt_pro.append(mpt.test(low_e_hitmap))
        if ECT:
            high_e_hitmap = sum(hitmaps[2:])
            ect_pro.append(ect.test_against(low_e_hitmap, high_e_hitmap))



    if MFT:
        nuc_vs_nuc = np.array(nuc_vs_nuc)
        nuc_vs_pro = np.array(nuc_vs_pro)
        pro_vs_nuc = np.array(pro_vs_nuc)
        pro_vs_pro = np.array(pro_vs_pro)

        save_result(nuc_vs_nuc, args.exposure, args.source_model, source_profile, "mfnuc6", args.input_directory)
        save_result(pro_vs_nuc, args.exposure, 0, source_profile, "mfnuc6", args.input_directory)

        save_result(nuc_vs_pro, args.exposure, args.source_model, source_profile, "mfpro6", args.input_directory)
        save_result(pro_vs_pro, args.exposure, 0, source_profile, "mfpro6", args.input_directory)

    if MFTI:
        nuc_vs_iso = np.array(nuc_vs_iso)
        pro_vs_iso = np.array(pro_vs_iso)

        save_result(nuc_vs_iso, args.exposure, args.source_model, source_profile, "mfiso6", args.input_directory)
        save_result(pro_vs_iso, args.exposure, 0, source_profile, "mfiso6", args.input_directory)

    if LVT:
        lvt_nuc = np.array(lvt_nuc)
        lvt_pro = np.array(lvt_pro)

        save_result(lvt_nuc, args.exposure, args.source_model, source_profile, f"lv{lvt_angle}", args.input_directory)
        save_result(lvt_pro, args.exposure, 0, source_profile, f"lv{lvt_angle}", args.input_directory)

    if MPT:
        mpt_nuc = np.array(mpt_nuc)
        mpt_pro = np.array(mpt_pro)

        save_result(mpt_nuc, args.exposure, args.source_model, source_profile, f"mp_e2", args.input_directory)
        save_result(mpt_pro, args.exposure, 0, source_profile, f"mp_e2", args.input_directory)

    if ECT:
        ect_nuc = np.array(ect_nuc)
        ect_pro = np.array(ect_pro)

        save_result(ect_nuc, args.exposure, args.source_model, source_profile, f"ec{ect_angle}_e2e4", args.input_directory)
        save_result(ect_pro, args.exposure, 0, source_profile, f"ec{ect_angle}_e2e4", args.input_directory)


    

if __name__ == "__main__":
    main()
