import math
import numpy as np
import scipy as sp
from scipy.special import exp1

from cosmology import *

MP = .938e9

# --- PROPAGATION - NUCLEI ---
def mfp_1(a, g):
    a_factor = 0.0095 * (a/56) ** (-3/2)
    lg = np.log10(g)
    return a_factor * np.where(lg > 9.5, 
                               np.exp(10 ** (10.3 - lg)), 
                               np.exp(10 ** 0.8) * 10  ** (-4 * (lg - 9.5)/3)
                        )

def d1(a, g, e0):
    return 2 * a * mfp_1(a, g) * (np.sqrt(a * MP * g / e0) - 1)

F = [0.0095, 2/2, 9.46, 10.31, 1.027] # F[4]=38/37
def mfp_c(a, g):
    a_factor = F[0] * np.power((a/56), -F[1])
    lg = np.log10(g)
    return a_factor * np.where(lg > F[2],
                               np.exp(np.power(10, F[3] - lg)),
                               np.exp(np.power(10, F[3] - F[2])) * np.power(10, -F[4] * (lg - F[2]))
                        )

def dc(a, g, e0):
    return a * mfp_c(a, g) * (np.log(a * g * MP) - np.log(e0))


MFP_EVOLUTION=-3-8/3

_zs = np.linspace(0, 1.5, 7500)
_deff = C / H0 * sp.integrate.cumulative_trapezoid((1 + _zs) ** (-1-MFP_EVOLUTION) /sqrtH(_zs), _zs, initial=0)

z2deff = sp.interpolate.interp1d(_zs, _deff)
deff2z = sp.interpolate.interp1d(_deff, _zs)

MFP_C_EVOLUTION=-3-2*F[4]

_zs = np.linspace(0, 1.5, 7500)
_dceff = C / H0 * sp.integrate.cumulative_trapezoid((1 + _zs) ** (-1-MFP_C_EVOLUTION) /sqrtH(_zs), _zs, initial=0)

z2dceff = sp.interpolate.interp1d(_zs, _dceff)
dceff2z = sp.interpolate.interp1d(_dceff, _zs)

def lossfactor(a, g, d):
    return np.power(1 + d / (2 * a * mfp_1(a, g)), -2)

def lossfactor_comovingd(a, g, d):
    d = z2deff(dprop2z(d))
    return np.power(1 + d / (2 * a * mfp_1(a, g)), -2)

def lossfactor_c(a, g, d):
    return np.exp(-d / mfp_c(a, g) / a)

def lossfactor_c_comovingd(a, g, d):
    d = z2dceff(dprop2z(d))
    return np.exp(-d / mfp_c(a, g) / a)


def d1_max(a, e0):
    g0 = e0 / (MP * a)
    if g0 < (5/8)**2 * (10 ** 9.5):
        return 0.34 * a * mfp_1(a, g0)
    if g0 < 10 ** 9.5:
        return ((10 ** 9.5 / g0) ** 0.5 - 1) * 2 * a * mfp_1(a, 10 ** 9.5)
    return g0 / (10 ** 10.3) * np.exp(-1) * a * mfp_1(a, g0)

def dndr(d, a, lowa, higha, gind, econvert, es, ds, FUCKYOU=False):
    # TODO maybe clean up some of the econvert stuff to make it more clear
    # STUPID
    startidx, stopidx = np.argmax(ds > d), len(ds) - 1 - np.argmax(ds[::-1] > d)
    if startidx == 0 and ds[0] <= d:
        return 0
    if startidx == stopidx:
        return 0 # Very dumb edge case lol, basically measure 0 so we'll ignore it
    
    if lowa != 0:
        lossfactors = lossfactor(a, es[startidx:stopidx] / a / MP, d) if not FUCKYOU else lossfactor_c(a, es[startidx:stopidx] / a / MP, d)
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
        print("Oh no")
        return None

def calc_cosmic_ray_rate_density(e0min, e0max, zs, source_model, FUCK2, FUCKYOU=False, lowa=0, higha=0):
    if source_model == 0: # TODO make this less hack-y
        #return calc_cosmic_ray_rate_density_proton(e0min, e0max, zs, get_source_parameters(source_model))
        return calc_cosmic_ray_rate_density_bonus(e0min, e0max, zs)

    gind, rc, zA, iA, j = get_source_parameters(source_model)

    dd = dc if FUCKYOU else d1

    ds = z2dceff(zs) if FUCKYOU else z2deff(zs) 

    if FUCK2 == 0:
        dndrs_borders = {14: [], 28: [], 56: []}
    else:
        dndrs_borders = {FUCK2: []}
    for i in range(2):
        e0 = [e0min, e0max][i]
        source_es = np.logspace(np.log10(e0), 21, 5000)
        for a in dndrs_borders.keys():
            if e0 == 1e23: # TODO
                dndrs_borders[a].append(0)
                continue

            econvert = zA[a] * rc
            source_ds = dd(a, source_es / (a * MP), e0)
            
            dndrs_borders[a].append(j * iA[a] * np.array([dndr(d, a, lowa, higha, gind, econvert, source_es, source_ds, FUCKYOU=FUCKYOU) / econvert for d in ds]))


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
        d = dc(a, g, e0)
        d[np.where(rs * z <= e0)] = 0
        _z = dceff2z(d)
        d = z2dprop(_z)
        ia = iA[a]
        spec = ia * np.exp(-rs/rc) * np.power(rs/rc, -gind) / z / rc / rc
        dist += spec * d
        dist[np.where(rs * z <= e0)] = 0
    return dist * j

# --- PROPAGATION - PROTONS ---
PROTON_EVOLUTION=-3

_zs = np.linspace(0, 0.5, 2500)
_deffp = C / H0 * sp.integrate.cumulative_trapezoid((1 + _zs) ** (-1-PROTON_EVOLUTION) /sqrtH(_zs), _zs, initial=0)

z2deffp = sp.interpolate.interp1d(_zs, _deffp)
deffp2z = sp.interpolate.interp1d(_deffp, _zs)

PROTON_PROP_VALUES = (np.float64(-0.17308559882366664), # alpha
                        np.float64(142565811523.5266),  # gamma_r
                        np.float64(4157908398652.891),  # c2
                        np.float64(54642255157.71955),  # gamma_c
                        np.float64(74638.91223478355))  # c1 = c2 * exp(gamma_r/gamma_c) * gamma_c^(-1-alpha) [to ensure continuity]

def atn_p(g):
    alpha, gammar, c2, gammac, c1 = PROTON_PROP_VALUES
    return np.where(g > gammac, 
                            c2 * np.exp(gammar / g) / g, 
                            c1 * np.power(g, alpha)                       
                    )


def dpro(e0, e):
    gamma0, gamma = e0 / MP, e / MP
    alpha, gammar, c2, gammac, c1 = PROTON_PROP_VALUES
    if gamma < gammac:
        return c1 / alpha * (np.power(gamma, alpha) - np.power(gamma0, alpha))
    if gamma0 > gammac:
        return c2 / gammar * (np.exp(gammar / gamma0) - np.exp(gammar / gamma))
    return c1 / alpha * (np.power(gammac, alpha) - np.power(gamma0, alpha)) + c2 / gammar * (np.exp(gammar / gammac) - np.exp(gammar / gamma))
     
def dndr_proton(d, gind, es, ds):
    startidx = np.argmax(ds > d)
    if startidx == 0 and ds[0] <= d:
        return 0
    starte = es[startidx]

    # Integral from starte to infinity of e ^ -gind: 
    return np.power(starte, 1 - gind) / gind

def calc_cosmic_ray_rate_density_proton(e0min, e0max, zs, source_model):

    gind, j = source_model # Source: dn/de = j * e ** -gind

    ds = z2deffp(zs)

    dndrs_borders = []
    for i in range(2):
        e0 = [e0min, e0max][i]

        source_es = np.logspace(np.log10(e0), 21, 5000)          
        source_ds = np.array([dpro(e0, e) for e in source_es])
        
        dndrs_borders.append(j * np.array([dndr_proton(d, gind, source_es, source_ds) for d in ds]))
    
    # No need to differentiate, we are looking at a range of energy so just subtraction is fine
    dndrs = dndrs_borders[0] - dndrs_borders[1]
    return dndrs


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
def calc_cosmic_ray_rate_density_bonus(e0min, e0max, zs):
    e0max = min(e0max, 4e20)
    e0s = np.linspace(e0min, e0max, num=int(np.ceil((e0max - e0min) / 5e15)))
    es = sp.integrate.solve_ivp(propa, [0, zs[-1]], e0s, t_eval=zs)
    dede = np.gradient(es.y, e0s, axis=0)

    # for i in range(0, len(e0s), len(e0s) // 10):
    #     plt.plot(zs, dede[i], label=f'{e0s[i]:.2e}')
    # plt.legend()
    # plt.show()
    # return 0

    dndrs = qev(es.y, zs) * dede

    return sp.integrate.cumulative_trapezoid(
        dndrs,
        e0s,
        axis=0
    )[-1]


# --- MODELS ---
def get_source_parameters(model):
    if model == 1:
        gind = 1
        fR = {4: 0.7, 14: 0.3, 28: 0.05, 56: 0} # Unclear value for helium here
        rc = 5e18
        q = 5e44
    elif model == 99:
        gind = -1.7
        fR = {4: 3.36, 14: 1, 28: 1/6, 56: 1/60}
        rc = 10 ** (18.2)
        q = 4e44
    elif model == -2:
        gind = -2
        fR = {4: 4, 14: 1, 28: 1/6, 56: 1/60} # Unclear values here too
        # fR = {4: 0.240 / 2, 14: 0.648 / 7, 28: 0.086 / 14, 56: 0.024 / 26}
        rc = 1.3e18 # at least I think
        q = 4e44
    elif model == -10:
        gind = -0.45
        rc = 1.6e18
        q = 4e44
        fR = {1: 0.2, 4: 0.5, 14: 0.26, 28: 0.027, 56: 0.013}
    elif model == 0: # PROTON WORK IN PROGRESS TODO

        return (2.2, 3e59) # gind, j
    else:
        print("Unknown model!")
        return
    
    zA = {1: 1, 4: 2, 14: 7, 28: 14, 56: 26}
    #fA   = {a: fR[a] / zA[a]**(1-gind) for a in fR.keys()}
    iA = {a: fR[a] * zA[a] / sum([fR[aa] * zA[aa] for aa in fR.keys()]) for a in fR.keys()}

    # erg -> eV
    q *= 624150907446
    j = q / (math.gamma(2-gind))

    # gind : power law index (number like -2 or 1)
    # rc : critical rigidity (V)
    # zA : charge of each nucleus
    # iA : fraction of each nucleus, when working in dimensionless units (x=E/ZRc)
    # j : normalization (to recreate q) (eV /yr /Mpc^3)
    # the spectrum, with these parameters, is exactly:
    #   dN/dx = j * iA / (Rc * zA) * x **-gind * exp(-x)
    # where x=E/(ZRc)
    return (gind, rc, zA, iA, j)


####### READY TO KRILL
from matplotlib import pyplot as plt
def main():
    sm = get_source_parameters(-2)
    e0s = np.logspace(19, 20.5, 100)

    acs = {56: 'blue', 28: 'orange', 14: 'green', 0: 'black'}
    for a in acs.keys():
        d90s = []
        for e0 in e0s:
            d90s.append(d90(e0, sm, a))
        plt.loglog(e0s, d90s, color=acs[a])
        
        d90s = []
        for e0 in e0s:
            d90s.append(d90(e0, sm, a, FUCKYOU=True))
        plt.loglog(e0s, d90s, color=acs[a], linestyle='--')

        # d90s = []
        # for e0 in e0s:
        #     d90s.append(d90(e0, sm, a, FUCKYOU=False, thre=0.5))
        # plt.loglog(e0s, d90s, color=acs[a], alpha=0.6)

        # d90s = []
        # for e0 in e0s:
        #     d90s.append(d90(e0, sm, a, FUCKYOU=True, thre=0.5))
        # plt.loglog(e0s, d90s, color=acs[a], linestyle='--', alpha=0.6)

    plt.grid(which='both')
    plt.show()

def d90(e0, sm, a, FUCKYOU=False, thre=0.9):
    dprops = np.logspace(-1, 3.5, 400)
    dndr = calc_cosmic_ray_rate_density(e0, 1e23, dprop2z(dprops), sm, a, FUCKYOU)
    sms = sp.integrate.cumulative_trapezoid(dndr, dprops)
    di = np.argmax(sms / sms[-1] > thre)
    return dprops[di]
    

def d2notd(d):
    return d / (1 + d * H0/ C)**MFP_EVOLUTION
notd2d = sp.interpolate.interp1d(d2notd(_deff), _deff, fill_value='extrapolate')
def d2(a, g, e0):
    return notd2d(d1(a, g, e0))

# redshift (d2 instead of d1)
# Different d choices (logspace and not linspace)
# change to proton mass
# different way of calculating integral
# ceiling energy vs no ceiling energy

"""def calc_cosmic_ray_rate_density2(e0min, e0max, zs, source_model):
     ...:     gind, rc, zA, iA, j = source_model
     ...:
     ...:     ds = z2dprop(zs)
     ...:
     ...:     #dndrs_borders = {14: [], 28: [], 56: []}
     ...:     dndrs_borders = {56: []}
     ...:     for i in range(2):
     ...:         e0 = [e0min, e0max][i]
     ...:         source_es = np.logspace(np.log10(e0), 21, 5000)
     ...:         for a in dndrs_borders.keys():
     ...:             if e0 == 1e23: # TODO
     ...:                 dndrs_borders[a].append(0)
     ...:                 continue
     ...:
     ...:             econvert = zA[a] * rc
     ...:             source_ds = d2(a, source_es / (a * MP), e0)
     ...:
     ...:             dndrs_borders[a].append(j * iA[a] * np.array([propagation.dndr(d, a, 1, a, gind, econvert, source_es, source_ds) / econvert for d in d     ...: s]))
     ...:
     ...:     # TODO
     ...:     # dndrs_borders[56][1] = 0
     ...:
     ...:     # No need to differentiate, we are looking at a range of energy so just subtraction is fine
     ...:     dndrs = sum([dndrs_borders[a][0] - dndrs_borders[a][1] for a in dndrs_borders.keys()])
     ...:     return dndrs"""

if __name__ == "__main__":
    main()
