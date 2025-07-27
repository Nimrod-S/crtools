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
def gpstime2date(gpstime):
    return datetime(1980, 1, 6) + timedelta(seconds=gpstime)

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
            "dec": float(event['sd_dec']), # TODO: add angular uncertainty (only has it with phi,theta which are a third coordinate system altogether)
            "eV": 1e18 * float(event['sd_energy']),
            "deV": 1e18 * float(event['sd_denergy']),
            "datetime": gpstime2date(int(event['gpstime']))
        })
        # To avoid duplicates (for multi eye events)
        found_ids.add(int(event['id']))

    return data

# --- FILTER ---
def energy_filter(data, emin, emax):
    # TODO consider energy uncertainty?
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
        elif coords == 'S':
            ph, th = event['ra'], event['dec']
            r = hp.Rotator(coord=['C', 'E'])
            yearpart = (event['datetime'].timetuple().tm_yday - 70) / 366
            rae, dece = r(ph, th, lonlat=True)
            rae -= yearpart * 360
            rae %= 360
            ph, th = r.get_inverse()(rae, dece, lonlat=True)
            
        ipix = hp.ang2pix(nside, ph, th, lonlat=True)
        hpmap[ipix] += 1

    return hpmap

# --- ANALYSIS ---
def timefun(data):
    vals = np.zeros(24)
    for event in data:
        vals[event['datetime'].hour - 4] += 1

    print(sum(vals))
    vals /= np.mean(vals)
    plt.xlabel("hour (UTC-4)")
    plt.ylabel("SD events")
    plt.bar(np.arange(24), vals)
    plt.show()
    return

    vals = np.zeros(12)
    for event in data:
        vals[event['datetime'].month - 1] += 1
    plt.bar(np.arange(12) + 1, vals)
    plt.show()

    for mi in range(12):
        hours = np.zeros(24)
        for event in data:
            if event['datetime'].month != mi + 1:
                continue
            hours[event['datetime'].hour] += 1    
        plt.bar(np.arange(24), hours)
        avg = hours * np.exp( 2j * np.pi / 24 * np.arange(24))
        normavg = sum(avg) / sum(hours)
        print(f"radius: {np.absolute(normavg)}, angle (hours): {np.angle(normavg) / np.pi * 12}")
        
        plt.show()


def get_ra_dist(data):
    ra_bins = np.linspace(0, 360, 13)
    ra_hits = np.zeros(len(ra_bins) - 1)
    
    for event in data:
        ra = event['ra']
        bi = np.argmax(ra < ra_bins) - 1
        ra_hits[bi] += 1

    return ra_bins, ra_hits


# --- MAIN ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-path", "-s", help="path for summary dir", default="/home/nimrod/physics/uhecr/summary")
    parser.add_argument("--nside", "-n", help="nside for the healpix map", default=NSIDE)
    return parser.parse_args()

def main():
    args = parse_args()
    at = exposure.create_exposure_map(args.nside, "auger10")

    data = parse_auger_summaries(args.summary_path)
    
    #data = energy_filter(data, 1e17, 2e18)
    
    #timefun(data)

    e0s = np.array([1.99526231e+19, 2.51188643e+19, 3.16227766e+19, 3.98107171e+19, 5.01187234e+19, 6.30957344e+19, 7.94328235e+19, 1.00000000e+20, 1.25892541e+20])
    mc = np.array([np.float64(492.1696847513242), np.float64(294.7179125638793), np.float64(158.46287049905817), np.float64(76.33455941464476), np.float64(37.94483494646917), np.float64(17.14897086813731), np.float64(6.410175544002524), np.float64(2.4837950744159345), np.float64(0.8028522554387274)])
    m1 = np.array([np.float64(594.131994734713), np.float64(345.1518678296589), np.float64(180.22902258835677), np.float64(82.17103580525409), np.float64(36.82234618578102), np.float64(16.284561248521555), np.float64(5.752093495295569), np.float64(2.0015459544391074), np.float64(0.6594072282306233)])
    mp = np.array([np.float64(437.7342013695336), np.float64(271.4583963172637), np.float64(159.71330801328358), np.float64(85.98524173056822), np.float64(40.60049362887794), np.float64(17.129356245332), np.float64(7.258670933232992), np.float64(2.9952714362746864), np.float64(1.3211015124305718)])
    mp2 = np.array([np.float64(453.85626291541803), np.float64(276.96854637761334), np.float64(160.34950092732188), np.float64(84.88641583751462), np.float64(39.344310215493834), np.float64(16.270395784940366), np.float64(6.720834683801316), np.float64(2.6991319035585835), np.float64(1.1612313426784788)])

    # 200
    mccp = np.array([np.float64(466.675116353509), np.float64(286.0249205522278), np.float64(166.74572191581296), np.float64(89.39945093027382), np.float64(42.53018961138608), np.float64(18.3866258659856), np.float64(7.915222973788245), np.float64(3.516342463146675), np.float64(1.668195696834236)])
    mcc = np.array([np.float64(523.1350665602826), np.float64(318.61147640504464), np.float64(173.78360391260264), np.float64(85.9239970840843), np.float64(41.998676129382), np.float64(19.258362493934087), np.float64(7.631237069788151), np.float64(2.862460418599843), np.float64(0.9437574630021643)])

    # 300
    mcccp = np.array([np.float64(456.2362146088663), np.float64(278.6517898881124), np.float64(161.53711723842673), np.float64(85.72479059896608), np.float64(39.93387279811738), np.float64(16.856250572786386), np.float64(7.254119203600331), np.float64(3.217167123619143), np.float64(1.523249811919774)])
    mccc = np.array([np.float64(500.4076034372488), np.float64(302.6598077614776), np.float64(165.8879630958088), np.float64(82.03115584785772), np.float64(39.842860247721994), np.float64(18.272999932910714), np.float64(7.267010800717844), np.float64(2.7102976239246788), np.float64(0.88783935227574)])

    ns = []
    for e0 in e0s:
        ns.append(len(energy_filter(data, e0, 2e21)))
    
    plt.xscale('log')
    plt.plot(e0s, mc, color='indianred', ls='--')
    plt.plot(e0s, mcc, color='indianred', ls=':')
    plt.plot(e0s, mccc, color='indianred', ls='-.')

    plt.plot(e0s, mp, color='mediumseagreen', ls='--')
    plt.plot(e0s, mccp, color='mediumseagreen', ls=':')
    plt.plot(e0s, mcccp, color='mediumseagreen', ls='-.')
    
    plt.plot(e0s, mp2, color='green', ls='--')
    plt.plot(e0s, m1, color='gray', ls='--')
    plt.scatter(e0s, ns, color='blue')
    plt.show()


    data = energy_filter(data, 2e19, 2e21)

    m = build_map(data, 'G', args.nside)
    data2 = energy_filter(data, 4e19, 2e21)
    m2 = build_map(data2, 'G', args.nside)
    
    print(sum(m))
    hp.mollview(m)
    plt.show()

    # print(analysis.test_smallvar(m, ))
    #print(f"!! {analysis.test_superg(m, 5, 55)}")

    # lvr = analysis.LocalVarianceTest1(at)
    # print(f"wow {lvr.test(m)}")
    sc = analysis.SmallCorrelationTest()
    print(f"sc: {sc.test_against(m, m2)}")

    lv2 = analysis.LocalVarianceTest2(gmf.deflection_random(4e18) * 180 / np.pi, hp.get_nside(m))
    print(f"local variance {lv2.test(m)}")
    lv2 = analysis.LocalVarianceTest2(20, hp.get_nside(m))
    print(f"local variance {lv2.test(m)}")
    lv2 = analysis.LocalVarianceTest2(30, hp.get_nside(m))
    print(f"local variance {lv2.test(m)}")
    lv2 = analysis.LocalVarianceTest2(40, hp.get_nside(m))
    print(f"local variance {lv2.test(m)}")

    mfnuc = analysis.BigMatchedFilterTest.load("/home/nimrod/physics/uhecr/mf/nuc")
    mfpro = analysis.BigMatchedFilterTest.load("/home/nimrod/physics/uhecr/mf/pro")
    print(f"mf nuc: {mfnuc.test(m)}")
    print(f"mf pro: {mfpro.test(m)}")

    mt = analysis.MatchTest(np.load("tmpavg.npy"), 45)
    hp.mollview(mt.test(m))
    mt = analysis.MatchTest(np.load("tmpavg.npy"), 90)
    hp.mollview(mt.test(m))
    mt = analysis.MatchTest(np.load("tmpavg.npy"), 20)
    hp.mollview(mt.test(m))

    out = analysis.OUTest(at, -70, 30, -80, -40, 45)
    # out = analysis.OUTest(at, -70, 30, -50, -40, 45)
    print(f"over/underdensity {out.test(m)}")

    r = hp.Rotator(rot=(137.37, 0, 83.68))
    hp.mollview(m)

    # rb, rh = get_ra_dist(data)
    # plt.figure()
    # plt.plot(rb[:-1], rh)
    plt.show()


if __name__ == "__main__":
    main()