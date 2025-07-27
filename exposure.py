import numpy as np
import healpy as hp

# --- EXPOSURE ---
class TerrestrialExposure:
    def __init__(self, lat, zenithmin, zenithmax):
        self._lat = lat
        self._zenithmin = zenithmin
        self._zenithmax = zenithmax

    def exposure_factor(self, dec):
        ximin = (np.cos(self._zenithmin) - np.sin(self._lat) * np.sin(dec)) / (np.cos(self._lat) * np.cos(dec))
        ximax = (np.cos(self._zenithmax) - np.sin(self._lat) * np.sin(dec)) / (np.cos(self._lat) * np.cos(dec))

        amin = np.pi if ximin < -1 else 0 if ximin > 1 else np.acos(ximin)
        amax = np.pi if ximax < -1 else 0 if ximax > 1 else np.acos(ximax)

        w = np.cos(self._lat) * np.cos(dec) * (np.sin(amax) - np.sin(amin)) + (amax - amin) * np.sin(self._lat) * np.sin(dec)
        return w #

def create_terrestrial_exposure_map(nside, lat, zenithmin, zenithmax):
    
    TE = TerrestrialExposure(lat, zenithmin, zenithmax)
    rot = hp.Rotator(coord=['G', 'C'])

    npix = hp.nside2npix(nside)
    map = np.zeros(npix)

    for ipix in range(npix):
        b, l = hp.pix2ang(nside, ipix)
        # The map we draw is galactic, hence the rotation
        dec, ra = rot(b, l)

        w = TE.exposure_factor(np.pi / 2 - dec)
        map[ipix] = w

    return map

def create_auger_exposure_map(nside, partial=False, new=False):
    auger_lat = -35.2 * np.pi / 180 # Probably accurate enough, Auger used it in the anisotropy paper
    
    vert_max_angle = 60 * np.pi / 180
    hori_max_angle = 80 * np.pi / 180
    basic_vert_map = create_terrestrial_exposure_map(nside, auger_lat, 0, vert_max_angle)
    basic_hori_map = create_terrestrial_exposure_map(nside, auger_lat, vert_max_angle, hori_max_angle)

    # IMPORTANT: this is to match the open data (latest event is in 2018), exposure increased since then
    total_exp_auger_km2yrsr_vert = 60400 # From the sd_exposure field of the open data. THIS IS THE FULL DATA, NOT JUST 10%
    total_exp_auger_km2yrsr_hori = 17700
    if new:
        # see "2022 report from the Auger-TA working group on UHECR arrival directions"
        total_exp_auger_km2yrsr_vert = 95700
        total_exp_auger_km2yrsr_hori = 26300
        # see "The Pierre Auger Observatory: Results and Prospects" (2025)
        # sum of both = 135000
    
    if partial:
        total_exp_auger_km2yrsr_vert /= 10
        total_exp_auger_km2yrsr_hori /= 10
    
    normalized_vert_map = basic_vert_map / np.sum(basic_vert_map) * total_exp_auger_km2yrsr_vert / hp.nside2pixarea(nside) # Pretty sure I'm not wrong about the angular stuff (because we want the map to have units km2yr and not km2yrsr)
    normalized_hori_map = basic_hori_map / np.sum(basic_hori_map) * total_exp_auger_km2yrsr_hori / hp.nside2pixarea(nside) # Pretty sure I'm not wrong about the angular stuff (because we want the map to have units km2yr and not km2yrsr)

    return normalized_vert_map + normalized_hori_map

def create_ta_exposure_map(nside):
    ta_lat = 39.3 * np.pi / 180 # TODO
    
    vert_max_angle = 55 * np.pi / 180 # TODO
    basic_map = create_terrestrial_exposure_map(nside, ta_lat, 0, vert_max_angle)

    # see "2022 report from the Auger-TA working group on UHECR arrival directions"
    total_exp_auger_km2yrsr = 18000
    total_exp_auger_km2yrsr /= 1
    normalized_map = basic_map / np.sum(basic_map) * total_exp_auger_km2yrsr / hp.nside2pixarea(nside) # Pretty sure I'm not wrong about the angular stuff (because we want the map to have units km2yr and not km2yrsr)

    return normalized_map

def create_isotropic_exposure_map(nside):
    npix = hp.nside2npix(nside)
    
    # For now, just a constant exposure of 2000 km2, 5 yr
    area = 20000
    time = 5

    at = np.ones(npix) * area * time / (4 * np.pi)

    # Units: [yr * km2]
    return at

def create_2022_exposure_map(nside):
    # see "2022 report from the Auger-TA working group on UHECR arrival directions"
    ta = create_ta_exposure_map(nside)
    auger = create_auger_exposure_map(nside, new=True)
    return ta + auger

# IMPORTANT note that this function returns an exposure map with units [yr * km^2], NOT [yr * km^2 * sr]! It is assumed that the exposure density is constant inside each pixel
def create_exposure_map(nside, exp):
    if exp == 'isotropic':
        return create_isotropic_exposure_map(nside)
    elif exp == 'auger':
        return create_auger_exposure_map(nside)
    elif exp == 'auger10':
        return create_auger_exposure_map(nside, partial=True)
    elif exp == 'ta':
        return create_ta_exposure_map(nside)
    elif exp == '2022':
        return create_2022_exposure_map(nside)
    else:
        print("unknown exposure pattern!!")
