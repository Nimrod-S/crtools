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
def load_hitmap(source_model, exposure, source_profile, bratio, name, root_path):
    path = root_path + "/maps"

    gind = source_model
    s0 = source_profile[0]
    logs0 = int(np.log10(s0))

    pattern = f"cr_{exposure}_{name}_m{gind}_s{logs0}_b{bratio}_"
    fnames = [fname for fname in os.listdir(path) if pattern in fname]
    
    fnames.sort(key=lambda fname:int(fname.split('_')[-1][:-4])) # sort by idx

    for fname in fnames:
        full_path = path + "/" + fname
        yield np.load(full_path).astype(np.int64) # We are going to start summing and averaging stuff, so back to serious numbers lol
    return

def save_result(data, exposure, source_model, source_profile, bratio, testname, root_path):
    path = root_path + "/results"
    gind = source_model
    s0 = source_profile[0]
    logs0 = int(np.log10(s0))


    full_name = f"{testname}_{exposure}_b{bratio}_m{gind}_s{logs0}"
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
    parser.add_argument("--bias", "-b", choices=['iso', 'neutral', 'high'], help="source distribution bias to 2MRS", default='neutral')
    parser.add_argument("--input-directory", "-i", help="path with randomize_rays.py results", default="./cr_output")

    parser.add_argument("--lvt", action="store_true")
    parser.add_argument("--ect", action="store_true")
    parser.add_argument("--dst", action="store_true")
    parser.add_argument("--mftc", action="store_true")
    parser.add_argument("--mftb", action="store_true")
    return parser.parse_args()

def main():

    args = parse_args()

    at = exposure.create_exposure_map(args.nside, args.exposure)
    bratio = {"iso": 0, "neutral": 1, "high": 1.7}[args.bias]

    source_profile = (np.power(10, args.source_density), args.source_evolution)
    es = np.linspace(2e19, 8e19, 7)
    es[-1] = 2e22
    iters = []
    iters_pro = []
    for e in es[:-1]:
        iters.append(iter(load_hitmap(args.source_model, args.exposure, source_profile, bratio, f"e{int(e / 1e19)}", args.input_directory)))
        iters_pro.append(iter(load_hitmap(0, args.exposure, source_profile, bratio, f"e{int(e / 1e19)}", args.input_directory)))


    MFTC = args.mftc
    MFTB = args.mftb
    LVT = args.lvt
    MPT = False
    ECT = args.ect
    DST = args.dst


    if MFTC:
        mfnuc = analysis.BigMatchedFilterTest.load(f"cr_output/patterns/mf_{args.exposure}_b1_e2_nuc")
        mfpro = analysis.BigMatchedFilterTest.load(f"cr_output/patterns/mf_{args.exposure}_b1_e2_pro")
        
        nuc_vs_nuc = []
        nuc_vs_pro = []
        pro_vs_nuc = []
        pro_vs_pro = []
    if MFTB:
        mflow = analysis.BigMatchedFilterTest.load(f"cr_output/patterns/mf_{args.exposure}_b0_e2_nuc")
        mfhigh = analysis.BigMatchedFilterTest.load(f"cr_output/patterns/mf_{args.exposure}_b1.7_e2_nuc")
        
        nuc_vs_low = []
        nuc_vs_high = []
        pro_vs_low = []
        pro_vs_high = []
    if LVT:
        lvt_angle = 16.5
        lvt = analysis.LocalVarianceTest(lvt_angle, args.nside)

        lvt_nuc = []
        lvt_pro = []
    if MPT:
        mpt_angle = 16.5
        mpt = analysis.MultipolesTest(mpt_angle, at)

        mpt_nuc = []
        mpt_pro = []
    if ECT:
        ect_angle = 16.5
        ect = analysis.SmallCorrelationTest(ect_angle, args.nside)

        ect_nuc = []
        ect_pro = []
    if DST:
        dst_angle = 16.5
        dst = analysis.MultipolesTest(dst_angle, at)
        
        dst_nuc = []
        dst_pro = []


    for hitmaps in tqdm.tqdm(zip(*iters), total=10000):
        low_e_hitmap = sum(hitmaps)

        if MFTC:
            nuc_vs_nuc.append(mfnuc.test(low_e_hitmap))
            nuc_vs_pro.append(mfpro.test(low_e_hitmap))
        if MFTB:
            nuc_vs_low.append(mflow.test(low_e_hitmap))
            nuc_vs_high.append(mfhigh.test(low_e_hitmap))
        if LVT:
            lvt_nuc.append(lvt.test(low_e_hitmap))
        if MPT:
            mpt_nuc.append(mpt.test(low_e_hitmap))
        if ECT:
            high_e_hitmap = sum(hitmaps[2:])
            ect_nuc.append(ect.test_against(low_e_hitmap, high_e_hitmap))
        if DST:
            high_e_hitmap = sum(hitmaps[2:])
            c1low = dst.other_test(low_e_hitmap)
            c1high = dst.other_test(high_e_hitmap)
            dst_nuc.append((c1low, c1high))



    for hitmaps in tqdm.tqdm(zip(*iters_pro), total=10000):
        low_e_hitmap = sum(hitmaps)

        if MFTC:
            pro_vs_nuc.append(mfnuc.test(low_e_hitmap))
            pro_vs_pro.append(mfpro.test(low_e_hitmap))
        if MFTB:
            pro_vs_low.append(mflow.test(low_e_hitmap))
            pro_vs_high.append(mfhigh.test(low_e_hitmap))
        if LVT:
            lvt_pro.append(lvt.test(low_e_hitmap))
        if MPT:
            mpt_pro.append(mpt.test(low_e_hitmap))
        if ECT:
            high_e_hitmap = sum(hitmaps[2:])
            ect_pro.append(ect.test_against(low_e_hitmap, high_e_hitmap))
        if DST:
            high_e_hitmap = sum(hitmaps[2:])
            c1low = dst.other_test(low_e_hitmap)
            c1high = dst.other_test(high_e_hitmap)
            dst_pro.append((c1low, c1high))



    if MFTC:
        nuc_vs_nuc = np.array(nuc_vs_nuc)
        nuc_vs_pro = np.array(nuc_vs_pro)
        pro_vs_nuc = np.array(pro_vs_nuc)
        pro_vs_pro = np.array(pro_vs_pro)

        save_result(nuc_vs_nuc, args.exposure, args.source_model, source_profile, bratio, "mfnuc", args.input_directory)
        save_result(pro_vs_nuc, args.exposure, 0, source_profile, bratio, "mfnuc", args.input_directory)

        save_result(nuc_vs_pro, args.exposure, args.source_model, source_profile, bratio, "mfpro", args.input_directory)
        save_result(pro_vs_pro, args.exposure, 0, source_profile, bratio, "mfpro", args.input_directory)

    if MFTB:
        nuc_vs_low = np.array(nuc_vs_low)
        nuc_vs_high = np.array(nuc_vs_high)
        pro_vs_low = np.array(pro_vs_low)
        pro_vs_high = np.array(pro_vs_high)

        save_result(nuc_vs_low, args.exposure, args.source_model, source_profile, bratio, "mflow", args.input_directory)
        save_result(pro_vs_low, args.exposure, 0, source_profile, bratio, "mflow", args.input_directory)

        save_result(nuc_vs_high, args.exposure, args.source_model, source_profile, bratio, "mfhig", args.input_directory)
        save_result(pro_vs_high, args.exposure, 0, source_profile, bratio, "mfhig", args.input_directory)

    if LVT:
        lvt_nuc = np.array(lvt_nuc)
        lvt_pro = np.array(lvt_pro)

        save_result(lvt_nuc, args.exposure, args.source_model, source_profile, bratio, f"lv{lvt_angle}", args.input_directory)
        save_result(lvt_pro, args.exposure, 0, source_profile, bratio, f"lv{lvt_angle}", args.input_directory)

    if MPT:
        mpt_nuc = np.array(mpt_nuc)
        mpt_pro = np.array(mpt_pro)

        save_result(mpt_nuc, args.exposure, args.source_model, source_profile, bratio, f"mp_e2", args.input_directory)
        save_result(mpt_pro, args.exposure, 0, source_profile, bratio, f"mp_e2", args.input_directory)

    if ECT:
        ect_nuc = np.array(ect_nuc)
        ect_pro = np.array(ect_pro)

        save_result(ect_nuc, args.exposure, args.source_model, source_profile, bratio, f"ec{ect_angle}_e2e4", args.input_directory)
        save_result(ect_pro, args.exposure, 0, source_profile, bratio, f"ec{ect_angle}_e2e4", args.input_directory)

    if DST:
        dst_nuc = np.array(dst_nuc)
        dst_pro = np.array(dst_pro)

        save_result(dst_nuc, args.exposure, args.source_model, source_profile, bratio, f"ds_e2e4", args.input_directory)
        save_result(dst_pro, args.exposure, 0, source_profile, bratio, f"ds_e2e4", args.input_directory)

    

if __name__ == "__main__":
    main()
