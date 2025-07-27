import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

def test(m):
    nside = hp.get_nside(m)

    londeg = -50
    latdeg = 30
    radrad = 15 * np.pi / 180

    londeg = -70; latdeg = 30; radrad = (40) * np.pi / 180

    vec = hp.ang2vec(londeg, latdeg, lonlat=True)
    selection_ipix_over = hp.query_disc(nside, vec, radrad)

    londeg = -55
    latdeg = -30
    radrad = 15 * np.pi / 180

    londeg = -80; latdeg = -40; radrad = (40) * np.pi / 180

    vec = hp.ang2vec(londeg, latdeg, lonlat=True)
    selection_ipix_under = hp.query_disc(nside, vec, radrad)

    phits_over = sum(m[selection_ipix_over])
    phits_under = sum(m[selection_ipix_under])

    #mean_selected_density = sum(at[selection_ipix_over]) / sum(at[selection_ipix_under])

    return phits_over / phits_under

def test_superg(m, width_deg, poles_deg):
    nside = hp.get_nside(m)
    # SUPERGALACTIC don't forget this
    r = hp.Rotator(rot=(137.37, 0, 83.68))

    selection_ipix_inside = []
    selection_ipix_outside = []
    for ipix in range(hp.nside2npix(nside)):
        th, ph = r(*hp.pix2ang(nside, ipix)) # Get angle of pix, rotate, take the theta, convert theta to latitude
        sgb = (np.pi / 2 - th) * 180 / np.pi
        sgl = ph * 180 / np.pi

        # if np.abs(sgl) < 100:
        #     continue

        if -width_deg < sgb < width_deg:
            selection_ipix_inside.append(ipix)
            continue
        if -poles_deg < sgb < poles_deg:
            continue
        selection_ipix_outside.append(ipix)

    selection_ipix_inside = np.array(selection_ipix_inside)
    selection_ipix_outside = np.array(selection_ipix_outside)
    
    mean_selected_density = 1.4477639858643865


    m[selection_ipix_inside] += .5
    # m[selection_ipix_under] -= 5
    hp.mollview(m)
    plt.show()


    phits = sum(m[selection_ipix_inside]) / sum(m[selection_ipix_outside])
    return phits / mean_selected_density

def test_smallvar(hitmap, at):
    nside = hp.npix2nside(len(hitmap))

    res = hp.nside2resol(nside) * 180 / np.pi

    goodpart = at > 1e-2 * np.mean(at)

    hmap = hitmap / (at + 1e-7)
    return np.std(hmap[goodpart])


def zoa_mask(nside):
    def in_zoa(ipix):
        lon, lat = hp.pix2ang(nside, ipix, lonlat=True)
        return (np.abs(lat) < 5) | ((np.abs(lat) < 10) & (np.abs((lon + 180) % 360 - 180) < 30))
    npix = hp.nside2npix(nside)
    return np.where(in_zoa(np.arange(npix)))


class ThinLonStripeTest:
    def __init__(self, at, lon, width):
        nside = hp.get_nside(at)
        
        self._selection = []
            
        for ipix in range(hp.nside2npix(nside)):
            ilon, ilat = hp.pix2ang(nside, ipix, lonlat=True)
            if np.abs((ilon - lon) % 360) < width:
                self._selection.append(ipix)
        
        self._mean_selected_density = sum(at[self._selection]) / sum(at)

    def test(self, hitmap):
        return sum(hitmap[self._selection]) / sum(hitmap) / self._mean_selected_density

class OUTest:
    def __init__(self, at, olon, olat, ulon, ulat, rad):
        nside = hp.get_nside(at)

        ovec = hp.ang2vec(olon, olat, lonlat=True)
        self._selection_over = hp.query_disc(nside, ovec, rad * np.pi / 180)

        uvec = hp.ang2vec(ulon, ulat, lonlat=True)
        self._selection_under = hp.query_disc(nside, uvec, rad * np.pi / 180)

        self._mean_selected_density = sum(at[self._selection_over]) / sum(at[self._selection_under])

    def test(self, hitmap):
        phits_over = sum(hitmap[self._selection_over]) 
        phits_under = sum(hitmap[self._selection_under])
        phits = phits_over / max(phits_under, 1e-20) # To avoid divide by zero
        if phits_under == 0:
            return np.nan
        return phits / self._mean_selected_density

class LocalVarianceTest1:
    def __init__(self, at):
        self._highres_nside = hp.get_nside(at)
        self._highres_npix = len(at)
        self._lowres_nside = 4 # TODO
        self._lowres_npix = hp.nside2npix(self._lowres_nside)
        self._lowres_factor = hp.nside2pixarea(self._lowres_nside) / hp.nside2pixarea(self._highres_nside)

        self._bad_idea_at = hp.ud_grade(at, self._lowres_nside)
        self._at = at
        
        def zoa(ipix):
            lon, lat = hp.pix2ang(self._lowres_nside, ipix, lonlat=True)
            return np.abs(lat) < 5 or (np.abs(lat) < 10 and np.abs((lon + 180) % 360 - 180) < 30)
        
        def highres_zoa(ipix):
            lon, lat = hp.pix2ang(self._highres_nside, ipix, lonlat=True)
            return np.abs(lat) < 5 or (np.abs(lat) < 10 and np.abs((lon + 180) % 360 - 180) < 30)
        
        self._highres_goodmask = np.where([not highres_zoa(i) and at[i] != 0 for i in np.arange(self._highres_npix)])
        self._good_mask = np.where([not zoa(i) and self._bad_idea_at[i] != 0 for i in np.arange(self._lowres_npix)])

    def test(self, hitmap):
        lowres_hitmap = hp.ud_grade(hitmap, self._lowres_nside, power=-2) # This power value conserves number-of-rays supposedly (otherwise the function takes the mean value)
        pixels = lowres_hitmap[self._good_mask] / self._bad_idea_at[self._good_mask]

        ##pixels /= np.sqrt()

        # plt.figure()
        # plt.hist(pixels, bins=np.linspace(0, 10 * np.mean(pixels)))
        # plt.show()
        # Zabito Boga
        pixels = lowres_hitmap / self._bad_idea_at
        ppixels = hp.ud_grade(hitmap / self._at, self._lowres_nside, power=-2)

        # hp.mollview(pixels)
        # hp.mollview(ppixels, title="pp")

        # print(f"hmm {np.std(ppixels[np.where((~np.isnan(ppixels) & (ppixels != hp.UNSEEN)))])}")
        # print(f"hmm2 {np.std(pixels[np.where((~np.isnan(pixels) & (pixels != hp.UNSEEN)))])}")

        return np.std(ppixels[np.where((~np.isnan(ppixels) & (ppixels != hp.UNSEEN)))])

class LocalVarianceTest2:
    def __init__(self, ang, nside):
        self._nside = nside
        self._npix = hp.nside2npix(nside)

        vecs = [hp.pix2vec(self._nside, ipix) for ipix in range(self._npix)]
        self._discs = [hp.query_disc(self._nside, v, ang * np.pi / 180) for v in vecs]

        self._mask = zoa_mask(nside)
        self._effective_area = 4 * np.pi - hp.nside2pixarea(nside) * len(self._mask)
        self._angrad = ang * np.pi / 180

    def test(self, hitmap):
        if len(hitmap) != self._npix:
            raise TypeError("WRONG SIZE HITMAP!")

        # Getting rid of the ZoA. Copy to not change the original mutable hitmap
        hitmap = hitmap.copy()
        hitmap[self._mask] = 0

        # hitmap[self._discs]
        # cuts = np.sum(hitmap[self._discs], axis=1) - 1
        # corrfun = sum(hitmap * cuts / 2)

        totalrays = 0
        corrfun = 0
        for ipix in range(self._npix):
            if hitmap[ipix] == 0:
                # Timesave
                continue

            totalrays += hitmap[ipix]
            # How many pairs?
            #   internals * (internals - 1) / 2 for internal-internal pairs
            #   internals * externals for internal-external pairs, but these ones will be counted twice (from the other pixels too), so actually internals * externals / 2
            # So in total, internals * (internals + externals - 1) / 2
            corrfun += hitmap[ipix] * (sum(hitmap[self._discs[ipix]]) - 1) / 2

        # Total amount of possible pairs is raycount * (raycount - 1) / 2, so we normalize TODO
        corrnorm = totalrays * (totalrays - 1) / 2

        return np.sqrt(corrfun / corrnorm * self._effective_area / np.pi) / self._angrad

class MatchTest:
    def __init__(self, avg, ang):
        self._nside = hp.get_nside(avg)
        self._npix = len(avg)
        self._avg = avg

        vecs = [hp.pix2vec(self._nside, ipix) for ipix in range(self._npix)]
        self._discs = [hp.query_disc(self._nside, v, ang * np.pi / 180) for v in vecs]
        
        self._avgtotal = sum(avg)
        self._avgvals = np.array([sum(avg[disc]) for disc in self._discs])

    def test(self, hitmap):
        total = sum(hitmap)
        vals = np.array([sum(hitmap[disc])for disc in self._discs])

        res = ((vals / total - self._avgvals / self._avgtotal) / (self._avgvals / self._avgtotal))
        res[np.where(self._avg == 0)] = hp.UNSEEN
        return res


class MatchedFilterTest:
    def __init__(self, signal, islog):
        self._nside = hp.get_nside(signal)
        self._npix = len(signal)
        norm = np.mean(signal)
        if islog:
            self._signal = np.log(signal / norm)
        else:
            self._signal = signal / norm


    def test(self, hitmap):
        return np.dot(hitmap, self._signal)

class BigMatchedFilterTest:
    def __init__(self, signal=None):
        if signal is None:
            return

        nside = hp.get_nside(signal)
        npix = len(signal)
        
        self._regions = [[], [], [], [], [], [], [], [], [], [], [], []]
        
        for ipix in range(npix):
            lon, lat = hp.pix2ang(nside, ipix, lonlat=True)
            # if (0 < lon < 90) and (lat < -10):
            #     self._regions[0].append(ipix)
            # if (90 < lon < 180) and (lat < -5):
            #     self._regions[1].append(ipix)
            # if (270 < lon) and (lat > 10):
            #     self._regions[2].append(ipix)
            # if (180 < lon < 270) and (lat > 5):
            #     self._regions[3].append(ipix)
            # if (180 < lon < 270) and (lat < -5):
            #     self._regions[4].append(ipix)
            # if (270 < lon) and (lat < -10):
            #     self._regions[5].append(ipix)
            # if (0 < lon < 90) and (lat > 10):
            #     self._regions[6].append(ipix)
            # if (90 < lon < 180) and (lat > 5):
            #     self._regions[7].append(ipix)
            if ((0 < lon < 30) and (lat < -10)) or (30 < lon < 120) and (lat < -5):
                self._regions[0].append(ipix)
            if (120 < lon < 150) and (lat < -5):
                self._regions[1].append(ipix)
            if (150 < lon < 180) and (lat < -5):
                self._regions[2].append(ipix)
            if (180 < lon < 210) and (lat < -5):
                self._regions[3].append(ipix)
            if (210 < lon < 240) and (lat < -5):
                self._regions[4].append(ipix)
            if ((330 < lon < 360) and (lat < -10)) or (240 < lon < 330) and (lat < -5):
                self._regions[5].append(ipix)
            if ((0 < lon < 30) and (lat > 10)) or (30 < lon < 120) and (lat > 5):
                self._regions[6].append(ipix)
            if (120 < lon < 150) and (lat > 5):
                self._regions[7].append(ipix)
            if (150 < lon < 180) and (lat > 5):
                self._regions[8].append(ipix)
            if (180 < lon < 210) and (lat > 5):
                self._regions[9].append(ipix)
            if (210 < lon < 240) and (lat > 5):
                self._regions[10].append(ipix)
            if ((330 < lon < 360) and (lat > 10)) or (240 < lon < 330) and (lat > 5):
                self._regions[11].append(ipix)
            

        norm = np.mean(signal)
        real_signal = np.log(signal / norm)
        
        self._signal_regions = []
        for region in self._regions:
            self._signal_regions.append(np.sum(real_signal[region]))
        self._signal_regions = np.array(self._signal_regions)


    def test(self, hitmap):
        hitmap_regions = []
        for region in self._regions:
            hitmap_regions.append(np.sum(hitmap[region]))
        hitmap_regions = np.array(hitmap_regions)
        return np.dot(hitmap_regions, self._signal_regions)

    def test_against(self, hitmap1, hitmap2):
        hitmap_regions1 = []
        hitmap_regions2 = []
        for region in self._regions:
            hitmap_regions1.append(np.sum(hitmap1[region]))
            hitmap_regions2.append(np.sum(hitmap2[region]))
        hitmap_regions1 = np.array(hitmap_regions1)
        hitmap_regions2 = np.array(hitmap_regions2)
        return np.dot(hitmap_regions1, hitmap_regions2)

    def save(self, path):
        np.savez(path, *self._regions, signal=self._signal_regions)

    @staticmethod
    def load(path):
        loaded = np.load(path + ".npz")
        mf = BigMatchedFilterTest()
        mf._signal_regions = loaded['signal']
        mf._regions = []
        for i in range(len(mf._signal_regions)):
            mf._regions.append(loaded[f'arr_{i}'])
        return mf
    
class SmallCorrelationTest:
    def __init__(self):
        pass
    
    def test_against(self, hitmap1, hitmap2):
        h1 = hp.sphtfunc.smoothing(hitmap1, sigma=1 * np.pi / 180)
        h2 = hp.sphtfunc.smoothing(hitmap2, sigma=1 * np.pi / 180)

        # ~6 degrees
        # h1 = hp.ud_grade(hitmap1, 16, power=-2)
        # h2 = hp.ud_grade(hitmap2, 16, power=-2)
        # hp.mollview(h1)
        # hp.mollview(h2)
        # hp.mollview(h3)
        # hp.mollview(h4)
        # plt.show()

        # proj = 0
        # norm1 = 0
        # norm2 = 0
        # for i in range(len(h1)):
        #     proj += h1[i] * h2[i]
        #     norm1 += h1[i] * h1[i]
        #     norm2 += h2[i] * h2[i]
        # return proj / np.sqrt(norm1 * norm2)
        return np.dot(h1, h2) / np.linalg.norm(h1) / np.linalg.norm(h2) #np.sqrt(np.dot(h1, h1) * np.dot(h2, h2))
    

class MultipolesTest:
    def __init__(self):
        pass

    def test(self, hitmap):
        # alm = hp.map2alm(hitmap)
        hitmap2 = hp.sphtfunc.smoothing(hitmap, sigma=20 * np.pi / 180)
        cl = hp.anafast(hitmap2)
        
        # if   h = h0 (1 + d cos (theta))
        # then c0 = 4pi * h0^2
        # and  c1 = 4pi * h0^2 d^2 / 9 (this 9 is (2l+1)^2)
        # so   d = 3 * sqrt(c1 / c0)
        return np.array([cl[1] / cl[0], cl[2] / cl[0]])

    # TODO do something w/ these functions
    def point(mp):
        al = hp.map2alm(mp, lmax=2)
        vec = np.array([np.real(al[3]) * np.sqrt(2), np.imag(al[3]) * np.sqrt(2), np.real(al[1])])
        return vec / np.real(al[0]) * np.sqrt(3)
    def mppoint(point, nside):
        npix = hp.nside2npix(nside)
        m = np.zeros(npix)
        for ipix in range(npix):
            v = hp.pix2vec(nside, ipix)
            m[ipix] = np.dot(v, point)
        return m
    