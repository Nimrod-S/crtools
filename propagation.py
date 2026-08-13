import argparse

import numpy as np
import scipy as sp
from scipy.special import exp1

import mfp
from cosmology import *

MP = .938e9

# --- PROPAGATION - NUCLEI ---

#F = [0.005, 2/2, 9.46, 10.31, 1.2] # F[4]=38/37
#F = [0.0095, 2/2, 9.46, 10.31, 1.027] # F[4]=38/37 OLD ATTEMPT
F = [0.0095, 2/2, 9.5, 10.3, 4/3]

# -- fully analytic --
def mfp_analytic(a, g):
    a_factor = F[0] * np.power((a/56), -F[1])
    lg = np.log10(g)
    return a_factor * np.where(lg > F[2],
                               np.exp(np.power(10, F[3] - lg)),
                               np.exp(np.power(10, F[3] - F[2])) * np.power(10, -F[4] * (lg - F[2]))
                        )

def d_analytic(a, g, e0):
    return a * mfp_analytic(a, g) * (np.log(a * g * MP) - np.log(e0))

def maxd(e0, a):
    thresh_low = 10**F[2] * a * MP / np.exp(1/F[4])
    thresh_high = 10**F[2] * a * MP / np.exp(10**(F[2]-F[3]))
    print(thresh_high)
    print(thresh_low)
    gm = np.where

    gm = np.select(
        [e0 < thresh_low, e0 < thresh_high, e0 >= thresh_high],
        [e0 * np.exp(1/F[4]) / (a * MP), 10**F[2], 10**F[2]] # TODO INCOMPLETE
    )

    return gm, d_analytic(a, gm, e0)


# -- semi analytic --
DP = mfp.DataParser("../CRPropa3/build/data")
def mfp_semi(a, g):
    return DP.mfp(a, g)

def d_semi(a, g, e0):
    return a * mfp_semi(a, g) * (np.log(a * g * MP) - np.log(e0))


MFP_EVOLUTION=-3-2*F[4]

_zs = np.linspace(0, 1.5, 7500)
_deff = C / H0 * sp.integrate.cumulative_trapezoid((1 + _zs) ** (-1-MFP_EVOLUTION) /sqrtH(_zs), _zs, initial=0)

z2deff = sp.interpolate.interp1d(_zs, _deff)
deff2z = sp.interpolate.interp1d(_deff, _zs)


def lossfactor(a, g, d):
    return np.exp(-d / mfp_semi(a, g) / a)

def lossfactor_comovingd(a, g, d):
    d = z2deff(dprop2z(d))
    return np.exp(-d / mfp_semi(a, g) / a)


def dndr(d, a, lowa, higha, gind, econvert, es, ds):
    startidx, stopidx = np.argmax(ds > d), len(ds) - 1 - np.argmax(ds[::-1] > d)
    if startidx == 0 and ds[0] <= d:
        return 0
    if startidx == stopidx:
        return 0 # Very dumb edge case, basically measure 0 so we'll ignore it
    
    if lowa != 0:
        lossfactors = lossfactor(a, es[startidx:stopidx] / a / MP, d)
        if lossfactors[-1] > (higha / a):
            return 0
        lowaidx, highaidx = np.argmin(lossfactors > (higha / a)), np.argmin(lossfactors > (lowa / a))
        if highaidx != 0 or lossfactors[-1] <= (lowa / a):
            stopidx = startidx + highaidx
        startidx = startidx + lowaidx
    
    starte, stope = es[startidx], es[stopidx]
    x1, x2 = starte / econvert, stope / econvert
    
    if gind == -2:
        # Now this is just a simple integral between e1 and e2 of something that is solvable
        return (x1**2 + 2*x1 + 2)*np.exp(-x1) - (x2**2 + 2*x2 + 2)*np.exp(-x2)
    elif gind == 1:
        return exp1(x1) - exp1(x2)
    else:
        return (sp.special.gammainc(1 - gind, x2) - sp.special.gammainc(1 - gind, x1)) * sp.special.gamma(1-gind)


def calc_cosmic_ray_rate_density(e0min, e0max, zs, source_model, lowa=0, higha=0):
    """
    Returns the volumetric emission rate of rays with observed energy e0min < E < e0max as a function of redshift

    e0min - lower observed energy
    e0max - upper observed energy
    zs - array of redshift bins
    source_model - -2 for nuclei, 0 for protons
    lowa - lower observed atomic mass (unbound if 0)
    higha - upper observed atomic mass
    """
    if source_model == 0: # TODO make this more elegant
        return calc_cosmic_ray_rate_density_protons(e0min, e0max, zs)

    gind, rc, zA, iA, j = get_source_parameters(source_model)

    ds = z2deff(zs)

    dndrs_borders = {14: [], 28: [], 56: []}

    for i in range(2):
        e0 = [e0min, e0max][i]
        source_es = np.logspace(np.log10(e0), 21, 5000)
        for a in dndrs_borders.keys():
            if e0 == 1e23:
                dndrs_borders[a].append(0)
                continue

            econvert = zA[a] * rc
            source_ds = d_semi(a, source_es / (a * MP), e0)
            
            dndrs_borders[a].append(j * iA[a] * np.array([dndr(d, a, lowa, higha, gind, econvert, source_es, source_ds) / econvert for d in ds]))
        

    # No need to differentiate, we are looking at a range of energy so just subtraction is fine
    dndrs = sum([dndrs_borders[a][0] - dndrs_borders[a][1] for a in dndrs_borders.keys()])
    return dndrs


def get_r_dist(e0, rs, source_model):
    # Distribution of rigidities rs above energy e0 (not normalized, roughly in counts /Mpc^2 /yr /sr) 
    dist = 0
    gind, rc, zA, iA, j = get_source_parameters(source_model)
    for a in [14, 28, 56]:
        z = zA[a]
        g = rs * z / MP
        d = d_semi(a, g, e0)
        d[np.where(rs * z <= e0)] = 0
        _z = deff2z(d)
        d = z2dprop(_z)
        ia = iA[a]
        spec = ia * np.exp(-rs/rc) * np.power(rs/rc, -gind) / z / rc / rc
        dist += spec * d
        dist[np.where(rs * z <= e0)] = 0
    return dist * j

def get_mean_r(es, source_model):
    if source_model == -2:
        rig = np.linspace(10 ** 17.5, 10 ** 19.4, 3000)
        rsa = [get_r_dist(e, rig, -2) for e in es]
        rsad = [rsa[i+1] - rsa[i] for i in range(len(es)-1)]
        raa = [np.sum(rig * rs) / np.sum(rs) for rs in rsad]
        return np.array(raa)
    elif source_model == 0:
        rsp = [e for e in es]
        rap = [(rsp[i+1] + rsp[i]) * 0.5 for i in range(len(es) -1)]
        return rap


# --- PROPAGATION - PROTONS ---
ECEP=2.7e18
T0EP=3.4e9
ECPI=3.2e20
T0PI=2.2e7

def xx(e):
    return 9.463/3.086e7 / (np.exp(-ECEP/e) / T0EP + np.exp(-ECPI/e) / T0PI) # numbers are for yr->mpc
def qev(e, z):
    return (np.power((e / 10 ** 19.6), -0.5) * np.power(1 + z, 3) * 0.6e44 * 624150907446 / e / e) / 1.7#* (e < 1e22)
def propa(z, e):
    return e * (1 / (1 + z) + C / (H0 * sqrtH(z) * (1 + z) * xx(e)))
def calc_cosmic_ray_rate_density_protons(e0min, e0max, zs):
    e0max = min(e0max, 1e21)
    e0s = np.linspace(e0min, e0max, num=int(np.ceil((e0max - e0min) / 5e15)))
    es = sp.integrate.solve_ivp(propa, [0, zs[-1]], e0s, t_eval=zs)
    dede = np.gradient(es.y, e0s, axis=0)

    dndrs = qev(es.y, zs) * dede

    return sp.integrate.cumulative_trapezoid(
        dndrs,
        e0s,
        axis=0
    )[-1]


# --- MODELS ---
def get_source_parameters(model):
    if model == -2:
        gind = -2
        fR = {4: 2 /2, 14: 6.2/7, 28: .9/14, 56: .16/26}
        rc = 10 ** 18.15
        q=5.85e44

    elif model == 0: # PROTON WORK IN PROGRESS

        return (2.2, 3e59) # gind, j
    else:
        print("Unknown model!")
        return
    
    zA = {1: 1, 4: 2, 14: 7, 28: 14, 56: 26}
    iA = {a: fR[a] * zA[a] / sum([fR[aa] * zA[aa] for aa in fR.keys()]) for a in fR.keys()}

    # erg -> eV
    q *= 624150907446
    j = q / (sp.special.gamma(2-gind))

    # gind : power law index (number like -2 or 1)
    # rc : critical rigidity (V)
    # zA : charge of each nucleus
    # iA : fraction of each nucleus, when working in dimensionless units (x=E/ZRc)
    # j : normalization (to recreate q) (eV /yr /Mpc^3)
    # the spectrum, with these parameters, is exactly:
    #   dN/dx = j * iA / (Rc * zA) * x **-gind * exp(-x)
    # where x=E/(ZRc)
    return (gind, rc, zA, iA, j)

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nside", "-n", help="nside for the healpix map", default=32)
    parser.add_argument("--output-directory", "-o", help="output path to save results in", default="/mnt/x/uhecr/cr_output")
    args = parser.parse_args()
    
    zs = np.linspace(0, 0.4, 401)[1:] # Important: resolution need to be better than the bias map voxel size

    es = np.linspace(2e19, 8e19, 7)
    es[-1] = 2e22

    es = np.array([20e18, 27.5e18, 32e18, 36.5e18, 42e18, 47.9e18, 50e18, 60e18, 70e18, 2e22]) # v2.0
    # es = np.array([1e20, 2e22]) # v3.0

    dndrs = []
    dndrs_pro = []

    for i in range(len(es) - 1):
        dndrs.append(calc_cosmic_ray_rate_density(es[i], es[i+1], zs, -2))
        dndrs_pro.append(calc_cosmic_ray_rate_density(es[i], es[i+1], zs, 0))

    dndrs = np.array(dndrs)
    dndrs_pro = np.array(dndrs_pro)
    
    np.save(args.output_directory+"/flux/energies_v2", np.array(es))
    np.save(args.output_directory+"/flux/flux_nuc_v2", dndrs)
    np.save(args.output_directory+"/flux/flux_pro_v2", dndrs_pro)


if __name__ == "__main__":
    main()
