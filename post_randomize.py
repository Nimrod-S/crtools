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
    parser.add_argument("--exposure", "-e", choices=['isotropic', 'auger', 'auger10', 'ta', '2022'], help="sky exposure pattern to use (default: isotropic)", default='isotropic')
    parser.add_argument("--source-density", "-sd", help="log10 source density (Mpc^-3)", type=float, default=-2.0)
    parser.add_argument("--source-evolution", "-se", help="source evolution index", type=float, default=0.0)
    parser.add_argument("--bias", "-b", choices=['iso', 'neutral', 'high'], help="source distribution bias to 2MRS", default='neutral')
    parser.add_argument("--input-directory", "-i", help="path with randomize_rays.py results", default="./cr_output")
    
    parser.add_argument("--gmf", "-g", type=int)
    parser.add_argument("--ecal", "-c", type=int, default=2)

    parser.add_argument("--lvt", action="store_true")
    parser.add_argument("--ect", action="store_true")
    parser.add_argument("--dst", action="store_true")
    parser.add_argument("--mftc", action="store_true")
    parser.add_argument("--mftb", action="store_true")
    parser.add_argument("--svt", action="store_true")
    return parser.parse_args()

def main():

    args = parse_args()

    at = exposure.create_exposure_map(args.nside, args.exposure)
    bratio = {"iso": 0, "neutral": 1, "high": 1.7}[args.bias]

    source_profile = (np.power(10, args.source_density), args.source_evolution, bratio)
    es = np.load(args.input_directory+"/flux/energies_v2.npy")

    magname = "" if args.gmf is None else f"_U{args.gmf}"

    iters_nuc = iter(load_hitmap(args.source_model, args.exposure, source_profile, f"v2"+magname, args.input_directory))
    iters_pro = iter(load_hitmap(0, args.exposure, source_profile, f"v2"+magname, args.input_directory))

    defmat = np.load(args.input_directory + f"/deflections/defmat_1kpc32{magname}.npy")

    MFTC = args.mftc
    MFTB = args.mftb
    LVT = args.lvt
    MPT = False
    ECT = args.ect
    DST = args.dst
    SVT = args.svt

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
        ect_angle = "D5"
        ect = analysis.SmallCorrelationTest(defmat, args.nside)

        ect_nuc = []
        ect_pro = []
    if DST:
        dst_angle = "D5"
        dst = analysis.MultipolesTest(defmat, at)
        
        dst_nuc = []
        dst_pro = []
    if SVT:
        compare_n = load_meanmap(-2, args.exposure, (1e-2, 0, 1), f"v2", args.input_directory)
        compare_p = load_meanmap(0, args.exposure, (1e-2, 0, 1), f"v2", args.input_directory)

        svt_angle = "D5"
        svt = analysis.SmallScaleVarTest(defmat, args.nside, np.sum(compare_n[args.ecal:], axis=0), np.sum(compare_p[args.ecal:], axis=0))

        svt_nuc_vs_nuc = []
        svt_nuc_vs_pro = []
        svt_pro_vs_nuc = []
        svt_pro_vs_pro = []


    #for hitmaps in tqdm.tqdm(zip(*iters), total=10000):
    for hitmaps in iters_nuc:
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
            lowmap = np.sum(hitmaps[args.ecal:args.ecal+2], axis=0)
            highmap = np.sum(hitmaps[args.ecal+2:], axis=0)
            ect_nuc.append(ect.test_against(lowmap, highmap))
        if DST:
            lowmap = np.sum(hitmaps[0:4], axis=0)
            highmap = np.sum(hitmaps[4:], axis=0)

            c1low = dst.other_test(lowmap)
            c1high = dst.other_test(highmap)
            dst_nuc.append((c1low, c1high))
        if SVT:
            midmap = np.sum(hitmaps[args.ecal:], axis=0)

            nn, pp = svt.test2(midmap)
            svt_nuc_vs_nuc.append(nn)
            svt_nuc_vs_pro.append(pp)



    #for hitmaps in tqdm.tqdm(zip(*iters_pro), total=10000):
    for hitmaps in iters_pro:

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
            lowmap = np.sum(hitmaps[args.ecal:args.ecal+2], axis=0)
            highmap = np.sum(hitmaps[args.ecal+2:], axis=0)
            ect_pro.append(ect.test_against(lowmap, highmap))
        if DST:
            lowmap = np.sum(hitmaps[0:4], axis=0)
            highmap = np.sum(hitmaps[4:], axis=0)

            c1low = dst.other_test(lowmap)
            c1high = dst.other_test(highmap)
            dst_pro.append((c1low, c1high))
        if SVT:
            midmap = np.sum(hitmaps[args.ecal:], axis=0)

            nn, pp = svt.test2(midmap)
            svt_pro_vs_nuc.append(nn)
            svt_pro_vs_pro.append(pp)


    if MFTC:
        nuc_vs_nuc = np.array(nuc_vs_nuc)
        nuc_vs_pro = np.array(nuc_vs_pro)
        pro_vs_nuc = np.array(pro_vs_nuc)
        pro_vs_pro = np.array(pro_vs_pro)

        save_result(nuc_vs_nuc, args.exposure, args.source_model, source_profile, "mfnuc", args.input_directory)
        save_result(pro_vs_nuc, args.exposure, 0, source_profile, "mfnuc", args.input_directory)

        save_result(nuc_vs_pro, args.exposure, args.source_model, source_profile, "mfpro", args.input_directory)
        save_result(pro_vs_pro, args.exposure, 0, source_profile, "mfpro", args.input_directory)

    if MFTB:
        nuc_vs_low = np.array(nuc_vs_low)
        nuc_vs_high = np.array(nuc_vs_high)
        pro_vs_low = np.array(pro_vs_low)
        pro_vs_high = np.array(pro_vs_high)

        save_result(nuc_vs_low, args.exposure, args.source_model, source_profile, "mflow", args.input_directory)
        save_result(pro_vs_low, args.exposure, 0, source_profile, "mflow", args.input_directory)

        save_result(nuc_vs_high, args.exposure, args.source_model, source_profile, "mfhig", args.input_directory)
        save_result(pro_vs_high, args.exposure, 0, source_profile, "mfhig", args.input_directory)

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

        txt = {2: "mid", 1: "low", 3: "high"}[args.ecal]

        save_result(ect_nuc, args.exposure, args.source_model, source_profile, f"ec{ect_angle}_{txt}"+magname, args.input_directory)
        save_result(ect_pro, args.exposure, 0, source_profile, f"ec{ect_angle}_{txt}"+magname, args.input_directory)

    if DST:
        dst_nuc = np.array(dst_nuc)
        dst_pro = np.array(dst_pro)

        save_result(dst_nuc, args.exposure, args.source_model, source_profile, f"ds_e2e42"+magname, args.input_directory)
        save_result(dst_pro, args.exposure, 0, source_profile, f"ds_e2e42"+magname, args.input_directory)

    if SVT:
        svt_nuc_vs_nuc = np.array(svt_nuc_vs_nuc)
        svt_nuc_vs_pro = np.array(svt_nuc_vs_pro)
        svt_pro_vs_nuc = np.array(svt_pro_vs_nuc)
        svt_pro_vs_pro = np.array(svt_pro_vs_pro)

        txt = {2: "mid", 1: "low", 3: "high"}[args.ecal]

        save_result(svt_nuc_vs_nuc, args.exposure, -2, source_profile, f"sv{svt_angle}_{txt}_nuc"+magname, args.input_directory)
        save_result(svt_pro_vs_nuc, args.exposure, 0, source_profile, f"sv{svt_angle}_{txt}_nuc"+magname, args.input_directory)
        save_result(svt_nuc_vs_pro, args.exposure, -2, source_profile, f"sv{svt_angle}_{txt}_pro"+magname, args.input_directory)
        save_result(svt_pro_vs_pro, args.exposure, 0, source_profile, f"sv{svt_angle}_{txt}_pro"+magname, args.input_directory)

    

if __name__ == "__main__":
    main()
