import numpy as np
import scipy as sp
import healpy as hp

from cosmology import *

from matplotlib import pyplot as plt


# --- LOCAL STRUCTURE ---
def parse_catalog_data(catalog_path):
    data = []
    with open(catalog_path) as f:
        for entry in f:
            fields = entry.split()
            data.append({
                "l": float(fields[3]),  # Galactic longitude (degrees)
                "b": float(fields[4]),  # Galactic latitude (degrees)
                "dL": float(fields[6]), # Luminosity distance (Mpc)
                "M": float(fields[8]),  # Solar masses (log10)
                "Mc": float(fields[10]),# Mass correction
                "SFR":float(fields[11]),# Star formation rate (solar masses per year)
                "Sc": float(fields[13]),# Star formation rate correction
            })
    return data

def selection_func_2mrs(z):
    A, alpha, gamma, zc = 116000, 2.108, 1.125, 0.025 # TODO citation
    return selection_func_generic(z, A, alpha, gamma, zc)

def selection_func_fit(z):
    A, alpha, gamma, zc = 1.72511352e+06, 1.63834331e+00, 1.67935457e+00, 2.38702796e-02
    return selection_func_generic(z, A, alpha, gamma, zc)

def selection_func_generic(z, A, alpha, gamma, zc):
    return A * np.power(z, gamma) * np.exp(-np.power(z / zc, alpha))

def lum_selection_func_fit(dl):
    A, alpha, dlc = .05074186, 2.46522663, 110.77033631159263
    return lum_selection_func_generic(dl, A, alpha, dlc)

def lum_selection_func_generic(dl, A, alpha, dlc):
    return A * sp.special.gammaincc(alpha, (dl / dlc)**2)

# Numbers from "Reconstructed density and velocity fields from the 2MASS Redshift Survey"
V_LG = 627
L_LG = 273
B_LG = 29
VEC_LG = hp.ang2vec(L_LG, B_LG, lonlat=True)
def fix_localgroup_v(l, b, v):
    vecgal = hp.ang2vec(l, b, lonlat=True)
    cos = np.dot(VEC_LG, vecgal)
    return v + cos * V_LG

def parse_catalog_data_mrs_pure(catalog_path):
    data = []
    with open(catalog_path) as f:
        for entry in f:
            if entry[0] == "#":
                continue
            fields = entry.split()

            data.append({
                "name": fields[-1],
                "l": float(fields[3]),  # Galactic longitude (degrees)
                "b": float(fields[4]),  # Galactic latitude (degrees)
                "v": float(fields[24]), # Redshift
                "Ks": float(fields[5])
            })

    return data

def parse_catalog_data_mrs(catalog_path, correction_path):
    corrections = {}
    with open(correction_path) as f:
        for line in f:
            fields = line.split()
            corrections[fields[0]] = float(fields[1])

    data = []
    with open(catalog_path) as f:
        for entry in f:
            if entry[0] == "#":
                continue
            fields = entry.split()

            if fields[0] in corrections:
                modc = corrections[fields[0]]
                z = np.power(10, modc / 5 - 5) * H0 / C
                # TODO extra bias to fix here
            else:
                real_v = fix_localgroup_v(float(fields[3]), float(fields[4]), float(fields[24]))
                z = max(real_v / C, 0)

            # if float(fields[5]) > 11.25:
            #     continue

            data.append({
                "name": fields[-1],
                "Ks": float(fields[5]),
                "l": float(fields[3]),  # Galactic longitude (degrees)
                "b": float(fields[4]),  # Galactic latitude (degrees)
                "z": z                  # Redshift
            })

    return data

def create_source_bias_voxel_map(catalog_data, halfsize, resolution):
    vcount = int(halfsize * 2 / resolution)
    grid = np.zeros((vcount, vcount, vcount))

    for entry in catalog_data:
        dl = entry['dL']
        if dl > halfsize: # TODO more fair treatment of the edge
            continue
        l, b = entry['l'] * np.pi / 180, entry['b'] * np.pi / 180
        gx, gy, gz =  dl * np.cos(l) * np.cos(b), dl * np.sin(l) * np.cos(b), dl * np.sin(b)
        x, y, z = int(gx / resolution + vcount / 2), int(gy / resolution + vcount / 2), int(gz / resolution + vcount / 2)

        grid[x][y][z] += np.power(10, entry['M']) / entry['Mc']
        
    # TODO sigma
    grid = sp.ndimage.gaussian_filter(grid, 2, mode="constant")*np.power(resolution,-3.) # I'm pretty sure this gets us density, so units are Solar Mass x Mpc^-3
    # grid *= np.power(resolution, -3.)

    # Finally, convert to bias... TODO
    grid /= np.mean(grid)

    return grid

def create_source_bias_map_old(catalog_path, nside, zs, max_catalog_dl):
    data = parse_catalog_data(catalog_path)

    voxel_resolution = 5 # In Mpc (dL). TODO rethink this number

    # It's easier to contain the data in a cartesian grid first, the voxels all have the same volume and it's easier to do smoothing
    grid = create_source_bias_voxel_map(data, max_catalog_dl, voxel_resolution)

    # Now, healpix-ify
    npix = hp.nside2npix(nside)
    hpmaps = np.zeros((len(zs), npix))

    voxel_count = grid.shape[0]
    for i, z in enumerate(zs):
        dl = z2dprop(z) * (1+z)

        for ipix in range(npix):
            
            if dl > max_catalog_dl:
                hpmaps[i][ipix] = 1
                continue
            vec = dl * np.array(hp.pix2vec(nside, ipix))
            xidx = int(voxel_count / 2 + vec[0] / voxel_resolution)
            yidx = int(voxel_count / 2 + vec[1] / voxel_resolution)
            zidx = int(voxel_count / 2 + vec[2] / voxel_resolution)
            
            hpmaps[i][ipix] = grid[xidx][yidx][zidx]    
    
    return hpmaps

def create_source_bias_map(catalog_path, nside, zs, min_catalog_dl, max_catalog_dl):
    """
    Below min_catalog_dl, assume there is nothing (basically cut out our sattelite dwarf galaxies)
    Above max_catalog_dl, assume isotropy
    """
    data = parse_catalog_data(catalog_path)

    dls = z2dprop(zs) * (1+zs)

    # Now, healpix-ify
    npix = hp.nside2npix(nside)
    sangle = hp.nside2pixarea(nside)
    hpmaps = np.zeros((len(zs), npix))
    galcounts = np.zeros(len(zs))
    totalmass = 0

    for entry in data:
        dl = entry['dL']
        if dl > max_catalog_dl or dl < min_catalog_dl:
            continue
        if dl > dls[-1]:
            print("PROBLEM")
            continue
        iz = np.argmax(dl <= dls)
        l, b = entry['l'], entry['b'] # deg
        ipix = hp.ang2pix(nside, l, b, lonlat=True)
        
        m = np.power(10, entry['M']) / entry['Mc']
        hpmaps[iz][ipix] += m
        totalmass += m

        galcounts[iz] += 1

    # Right now, hpmaps is not volume-normalized and does not represent density, just mass count. We will deal with this later during normalization.
    normalized = False
    for i, dl in enumerate(dls):
        if dl <= max_catalog_dl and galcounts[i] != 0:
            # Smooth:
            lmean = (4 * np.pi * dl ** 2 / galcounts[i]) ** (1/2) # Roughly mean distance, in mpc (suspicious geometric factor)
            lmean = max(lmean, 10)

            hpmaps[i] = hp.sphtfunc.smoothing(hpmaps[i], sigma=lmean / dl)
            # hpmaps[i] = hp.sphtfunc.smoothing(hpmaps[i], sigma=17 * np.pi / 180)

            # Rarely, numerical artifacts of the smoothing produces some negative values. From experience they are all <1e-4 of the mean so this is probably not that bad. We HAVE to fix it or the poisson distributions will get messed up later. TODO
            hpmaps[i] = np.maximum(hpmaps[i], 0)

            hpmaps[i] /= dl ** 2 * sangle

        if dl > max_catalog_dl:
            # We are out, now normalize and start being isotropic:
            if not normalized:
                normalized = True
                hpmaps[:i] *= np.sum(4 * np.pi * dls[:i] ** 2) / totalmass # I thought about this a lot and wrote stuff on paper lol
        
            # Just fill with isotropic value
            hpmaps[i] = np.ones(npix)
            continue

    return hpmaps

def fill_galactic_plane(catalog_data, min_z, max_z):
    # Based on: Crook et al. 2007 (arXiv:astro-ph/0610732)
    # The longitude of a bin represents the smaller edge
    lon_bins = np.linspace(0, 360, 37)
    lat_internal = np.array(3*[10] + 30*[5] + 3*[10])
    lat_external = np.array(3*[20] + 30*[15] + 3*[20])
    mean_factor = np.array(3*[1] + 30*[0.5] + 3*[1])
    z_bins = np.linspace(min_z, max_z, int((max_z - min_z) / 0.0033)) # Rough number choice
    counts_internal = np.zeros((len(z_bins) - 1, 36)) # The ones we already see inside the plane
    counts_external = np.zeros((len(z_bins) - 1, 36))
    magnitudes_external = [[[] for _ in range(36)] for _ in range(len(z_bins) - 1)]

    for entry in catalog_data:
        z = entry['z']
        if z > max_z or z < min_z:
            continue
        zbin = np.argmin(z > z_bins) - 1
        
        l, b = entry['l'], entry['b']
        lbin = np.argmin(l > lon_bins) - 1

        if np.abs(b) > lat_external[lbin]:
            continue
        if np.abs(b) > lat_internal[lbin]:
            counts_external[zbin][lbin] += 1
            magnitudes_external[zbin][lbin].append(entry['Ks'])
        else:
            counts_internal[zbin][lbin] += 1

    # Cosine is a geometrical factor since the upper latitudes are effectively a little smaller
    missing_counts = counts_external * mean_factor / np.cos(np.pi/180 * lat_internal) - counts_internal
    missing_N = np.where(counts_external != 0,
        sp.stats.norm(loc=missing_counts, scale=1).rvs(),
        0
    ) # TODO CHANGE

    for li in range(len(lon_bins) - 1):
        for zi in range(len(z_bins) - 1):
            N = int(missing_N[zi][li] + 0.5)
            if N <= 0:
                continue
            ls = sp.stats.uniform(loc=lon_bins[li], scale=lon_bins[li + 1]-lon_bins[li]).rvs(size=N)
            bs = sp.stats.uniform(loc=-lat_internal[li], scale=lat_internal[li]*2).rvs(size=N)
            zs = sp.stats.uniform(loc=z_bins[zi], scale=z_bins[zi + 1]-z_bins[zi]).rvs(size=N)
            ks = np.random.choice(magnitudes_external[zi][li], N) # replace=True. TODO CHANGE
            for l, b, z, k in zip(ls, bs, zs, ks):
                catalog_data.append({
                    "name": "FAKE",
                    "l": l,  # Galactic longitude (degrees)
                    "b": b,  # Galactic latitude (degrees)
                    "z": z,  # Redshift
                    "Ks": k     # K band magnitude
                })


def create_source_bias_map_mrs(catalog_path, correction_path, nside, zs, min_catalog_dl, max_catalog_dl, fill_plane=True):
    """
    Below min_catalog_dl, assume there is nothing (basically cut out our sattelite dwarf galaxies)
    Above max_catalog_dl, assume isotropy
    """
    data = parse_catalog_data_mrs(catalog_path, correction_path)

    dls = z2dprop(zs) * (1+zs)

    npix = hp.nside2npix(nside)
    sangle = hp.nside2pixarea(nside)
    hpmaps = np.zeros((len(zs), npix))
    galcounts = np.zeros(len(zs))
    totalmass = 0

    if (fill_plane):
        fill_galactic_plane(data, min_catalog_dl * H0 / C, max_catalog_dl * H0 / C)

    # TODO KILL
    vecs = np.zeros((len(dls), 3))

    # Build a healpy map
    print(len(data))
    for entry in data:
        z = entry['z']
        dl = z2dprop(z) * (1 + z)
        if dl > max_catalog_dl or dl < min_catalog_dl:
            continue
        if dl > dls[-1]:
            print("PROBLEM")
            continue
        iz = np.argmax(dl <= dls)
        l, b = entry['l'], entry['b'] # deg
        ipix = hp.ang2pix(nside, l, b, lonlat=True)
        
        m = 1 / selection_func_fit(z) * dl**2 / (1+z) ** 4 # We use da=dl/(1+z)2 for volume element
        hpmaps[iz][ipix] += m
        totalmass += m

        galcounts[iz] += 1

        # TODO kill
        vec = hp.ang2vec(l, b, lonlat=True) * m / (4 * np.pi * dl ** 2)
        for i in range(iz, len(dls)):
            vecs[i] += vec
    

    vecs *= 200**3 * np.pi * 4 / 3 / totalmass
    rdlen2 = np.linalg.norm(vecs, axis=1)
    # plt.loglog(dls, selection_func_2mrs(zs))
    # plt.loglog(dls, selection_func_fit(zs))
    # plt.figure()
    # plt.plot(dls, rdlen2)
    # plt.plot(dls, vecs[...,0], linestyle='--')
    # plt.plot(dls, vecs[...,1], linestyle='--')
    # plt.plot(dls, vecs[...,2], linestyle='--')
    # plt.show()
        

        
    # Right now, hpmaps is not volume-normalized and does not represent density, just mass count. We will deal with this later during normalization.
    normalized = False
    for i, dl in enumerate(dls):
        if dl <= max_catalog_dl and galcounts[i] != 0:
            # Smooth:
            lmean = (4 * np.pi * dl ** 2 / galcounts[i]) ** (1/2) # Roughly mean distance, in mpc (suspicious geometric factor)
            lmean = max(lmean, 10)

            hpmaps[i] = hp.sphtfunc.smoothing(hpmaps[i], sigma=lmean / dl)

            # Rarely, numerical artifacts of the smoothing produces some negative values. From experience they are all <1e-4 of the mean so this is probably not that bad. We HAVE to fix it or the poisson distributions will get messed up later. TODO
            hpmaps[i] = np.maximum(hpmaps[i], 0)

            hpmaps[i] /= dl ** 2 * sangle
            
        if dl > max_catalog_dl:
            # We are out, now normalize and start being isotropic:
            if not normalized:
                normalized = True
                hpmaps[:i] *= np.sum(4 * np.pi * dls[:i] ** 2) / totalmass # I thought about this a lot and wrote stuff on paper lol
        
            # Just fill with isotropic value
            hpmaps[i] = np.ones(npix)
            continue

    vecs = np.array(hp.pix2vec(nside, np.arange(npix)))
    delta = hpmaps - 1

    wow = []
    for i in range(delta.shape[0]):
        vec = np.sum(vecs * delta[i] / (4 * np.pi), axis=1) * sangle * np.gradient(dls)[i]
        wow.append(vec)
    wow = np.array(wow)
    wowr = np.cumsum(wow, axis=0)

    # plt.plot(dls, selection_func_fit(zs) / dls**2)

    rdlen = np.linalg.norm(wow, axis=1)
    rdlen2 = np.linalg.norm(wowr, axis=1)
    # plt.figure()
    # plt.plot(dls, rdlen)
    # plt.plot(dls, rdlen2)
    # plt.plot(dls, wowr[...,0], linestyle='--')
    # plt.plot(dls, wowr[...,1], linestyle='--')
    # plt.plot(dls, wowr[...,2], linestyle='--')
    print(wowr[30])
    print(hp.vec2ang(wowr[30], lonlat=True))
    def mppoint(point, nside):
        npix = hp.nside2npix(nside)
        m = np.zeros(npix)
        for ipix in range(npix):
            v = hp.pix2vec(nside, ipix)
            m[ipix] = np.dot(v, point)
        return m
    # hp.mollview(mppoint(wowr[30], nside))
    print(dls[30])
    grp1 = np.sum(hpmaps[0:30], axis=0)
    # hp.mollview(grp1)
    c = hp.anafast(grp1)
    print(c[1] / c[0])
    print(3 * np.sqrt(c[1] / c[0]))


    dlims = np.array([10, 50, 100, 200, 300])
    lims = [0] + [np.argmax(dlim <= dls) for dlim in dlims]

    hmaps = hpmaps.copy()
    for i, dl in enumerate(dls):
        pass

    # plt.figure()
    # plt.loglog(dls, np.sum(hmaps, axis=1) * dls ** 2)
    # plt.show()
        # hmaps[i] /= dl**2
    c = []
    grp1 = np.sum(hmaps[lims[0]:lims[1]], axis=0)
    c.append(hp.anafast(grp1))
    grp2 = np.sum(hmaps[lims[1]:lims[2]], axis=0)
    c.append(hp.anafast(grp2))
    grp3 = np.sum(hmaps[lims[2]:lims[3]], axis=0)
    c.append(hp.anafast(grp3))
    grp4 = np.sum(hmaps[lims[0]:lims[2]], axis=0)
    c.append(hp.anafast(grp4))
    grp5 = np.sum(hmaps[lims[0]:lims[3]], axis=0)
    c.append(hp.anafast(grp5))
    grp6 = np.sum(hmaps[lims[3]:lims[4]], axis=0)
    c.append(hp.anafast(grp6))
    grp7 = np.sum(hmaps[lims[0]:lims[4]], axis=0)
    c.append(hp.anafast(grp7))
    grp8 = np.sum(hmaps[lims[4]:lims[5]], axis=0)
    c.append(hp.anafast(grp8))

    for cc in c:
        print(f"c1: {cc[1] / cc[0]}, d: {3 * np.sqrt(cc[1] / cc[0])}")
    print("END")



    return hpmaps


def create_source_bias_map_mrsl(catalog_path, correction_path, nside, zs, min_catalog_dl, max_catalog_dl, fill_plane=True):
    """
    Below min_catalog_dl, assume there is nothing (basically cut out our satellite dwarf galaxies)
    Above max_catalog_dl, assume isotropy
    """
    data = parse_catalog_data_mrs(catalog_path, correction_path)

    dls = z2dprop(zs) * (1+zs)

    npix = hp.nside2npix(nside)
    sangle = hp.nside2pixarea(nside)
    hpmaps = np.zeros((len(zs), npix))
    galcounts = np.zeros(len(zs))
    totallum = 0

    if (fill_plane):
        fill_galactic_plane(data, min_catalog_dl * H0 / C, max_catalog_dl * H0 / C)

    # Build a healpy map
    for entry in data:
        z = entry['z']
        dl = z2dprop(z) * (1 + z)
        if dl > max_catalog_dl or dl < min_catalog_dl:
            continue
        if dl > dls[-1]:
            print("PROBLEM")
            continue
        iz = np.argmax(dl <= dls)
        l, b = entry['l'], entry['b'] # deg
        ipix = hp.ang2pix(nside, l, b, lonlat=True)
        
        k = -6 * np.log(1 + z) # k-correction
        l = np.power(10, -0.4 * (entry['Ks'] - k)) * dl**2 # The "actual" luminosity in useful units (erg/s for example) is of course some factor times this but we are going to normalize it anyway.
        l /= lum_selection_func_fit(dl)
        hpmaps[iz][ipix] += l
        totallum += l

        galcounts[iz] += 1 

        
    # Right now, hpmaps is not volume-normalized and does not represent density, just total luminosity. We will deal with this later during normalization.
    normalized = False
    for i, dl in enumerate(dls):
        if dl <= max_catalog_dl and galcounts[i] != 0:
            # Smooth:
            lmean = (4 * np.pi * dl ** 2 / galcounts[i]) ** (1/2) # Roughly mean distance, in mpc (suspicious geometric factor)
            lmean = max(lmean, 10)

            hpmaps[i] = hp.sphtfunc.smoothing(hpmaps[i], sigma=lmean / dl)

            # Rarely, numerical artifacts of the smoothing produces some negative values. From experience they are all <1e-4 of the mean so this is probably not that bad. We HAVE to fix it or the poisson distributions will get messed up later. TODO
            hpmaps[i] = np.maximum(hpmaps[i], 0)

            hpmaps[i] /= dl ** 2 * sangle
            
        if dl > max_catalog_dl:
            # We are out, now normalize and start being isotropic:
            if not normalized:
                normalized = True
                hpmaps[:i] *= np.sum(4 * np.pi * dls[:i] ** 2) / totallum # I thought about this a lot and wrote stuff on paper lol
        
            # Just fill with isotropic value
            hpmaps[i] = np.ones(npix)
            continue

    return hpmaps