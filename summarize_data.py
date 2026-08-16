import math
import csv
import numpy as np
import scipy as sp
import tqdm
from matplotlib import pyplot as plt
import healpy as hp
import argparse
from datetime import datetime, timedelta

import exposure
import analysis
import gmf

NSIDE=32

# --- PARSING ---
def parse_auger_summaries(path):
    raw_data = []
    for fname in ['dataSummarySD1500.csv', 'dataSummaryInclined.csv']:
        full_name = path + '/' + fname
        f = open(full_name)
        dr = csv.DictReader(f)
        raw_data += list(dr)
        f.close()

    data = []
    found_ids = set()
    # Now running a filter and converting to a more convenient format
    for event in raw_data:
        if event['sd1500'] == '0':
            continue
        elif int(event['id']) in found_ids:
            continue

        data.append({
            "id": int(event['id']),
            "b": float(event['sd_b']),
            "l": float(event['sd_l']),
            "ra": float(event['sd_ra']),
            "dec": float(event['sd_dec']), # FUTURE: add angular uncertainty
            "eV": 1e18 * float(event['sd_energy']),
            "deV": 1e18 * float(event['sd_denergy']),
        })
        # To avoid duplicates (for multi eye events)
        found_ids.add(int(event['id']))

    return data

def parse_uhecr_data(path):
    full_name = path + '/' + "AugerApJS2022_Yr_JD_UTC_Th_Ph_RA_Dec_E_Expo.dat"
    f = open(full_name)
    lines = f.readlines()[1:] # Skip the header
    f.close()

    data = []
    r = hp.Rotator(coord=['C', 'G'])
    for line in lines:
        fields = line.split()
        data.append({
            "id": int(0),
            # "deV": 1e18 * float(event['sd_denergy']),
            "ra": float(fields[5]),
            "dec": float(fields[6]), # FUTURE: add angular uncertainty
            "eV": 1e18 * float(fields[7]),
        })
        ra, dec = data[-1]['ra'], data[-1]['dec']
        l, b = r(ra, dec, lonlat=True)
        data[-1]['l'], data[-1]['b'] = l, b

    return data

# --- FILTER ---
def energy_filter(data, emin, emax):
    # FUTURE consider energy uncertainty?
    filtered = [event for event in data if emin < event['eV'] < emax]
    return filtered

# --- MAP ---
def build_map(data, coords, nside):
    npix = hp.nside2npix(nside)
    hpmap = np.zeros(npix)

    for event in data:
        if coords == 'G':
            ph, th = event['l'], event['b']
        elif coords == 'E':
            ph, th = event['ra'], event['dec']
            
        ipix = hp.ang2pix(nside, ph, th, lonlat=True)
        hpmap[ipix] += 1

    return hpmap


# --- MAIN ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-path", "-s", help="path for summary dir", default="../summary")
    parser.add_argument("--dataf-path", "-f", help="path for data dir", default="../fdata")
    parser.add_argument("--nside", "-n", help="nside for the healpix map", default=NSIDE)
    return parser.parse_args()

def main():
    args = parse_args()

    # NOTE these are used for the Auger open data (much less data available)
    # at = exposure.create_exposure_map(args.nside, "auger10")
    # data = parse_auger_summaries(args.summary_path)

    at = exposure.create_exposure_map(args.nside, "auger")
    data = parse_uhecr_data(args.dataf_path)
    datahigh = energy_filter(data, 42e18, 1e22)

    m = build_map(data, 'G', args.nside)
    mhigh = build_map(datahigh, 'G', args.nside)

    hp.mollview(m)

    print(f"TOTAL RAY COUNT: {sum(m)}")

    mask = 10
    magangle = 16.7

    nuc_signal = np.array([0.9302185,  0.30809623, 0.56556634, 1.99914073, 1.26947052])
    pro_signal = np.array([0.63815697, 0.37719293, 0.75388441, 1.9426459,  1.22234523])
    mft_nuc = analysis.BigMatchedFilterTest(mask, args.nside, signal_vector=nuc_signal)
    mft_pro = analysis.BigMatchedFilterTest(mask, args.nside, signal_vector=pro_signal)
    ent = analysis.SmallScaleVarTest(magangle, args.nside, mask)
    ect = analysis.SmallCorrelationTest(magangle, mask, args.nside)
    mt = analysis.MultipolesTest(at)

    print("LARGE SCALE CORRELATION (w/ nuclei, w/ protons, nuc - pro):")
    mn, mp = mft_nuc.test(m), mft_pro.test(m)
    print(mn, mp, mn - mp)

    print("ENTROPY (regular, effective):")
    print(ent.test_ent(m))

    print("ENERGY CORRELATION (32 EeV & 42 EeV):")
    print(ect.test_against(m, mhigh))

    print("DIPOLE (vector, amplitude):")
    d = mt.dipole_semiexp(m)
    print(d, np.linalg.norm(d))

    plt.show()
    return


if __name__ == "__main__":
    main()
