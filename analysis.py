import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def zoa_mask(nside):
    def in_zoa(ipix):
        lon, lat = hp.pix2ang(nside, ipix, lonlat=True)
        return (np.abs(lat) < 5) | ((np.abs(lat) < 10) & (np.abs((lon + 180) % 360 - 180) < 30))
    npix = hp.nside2npix(nside)
    return np.where(in_zoa(np.arange(npix)))


class LocalVarianceTest:
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
            

        # norm = np.mean(signal)
        # real_signal = np.log(signal / norm)
        
        # self._signal_regions = []
        # for region in self._regions:
        #     self._signal_regions.append(np.sum(real_signal[region]))
        # self._signal_regions = np.array(self._signal_regions)


        norm_signal = signal / np.mean(signal)

        self._signal_regions = []
        for region in self._regions:
            self._signal_regions.append(np.sum(norm_signal[region]))
        self._signal_regions = np.array(self._signal_regions)

        self._signal_regions = np.log(self._signal_regions)
        


    def test(self, hitmap):
        hitmap_regions = []
        for region in self._regions:
            hitmap_regions.append(np.sum(hitmap[region]))
        hitmap_regions = np.array(hitmap_regions)
        sm = np.sum(hitmap_regions)
        if 0 == sm:
            return 0
        return np.dot(hitmap_regions, self._signal_regions) / sm

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
    
# Consider: import s2fft
class SmallCorrelationTest:
    def __init__(self, ang, nside):
        self._mask = zoa_mask(nside)
        self._angle = ang
        pass
    
    def test_against(self, hitmap1, hitmap2):
        h1 = hp.sphtfunc.smoothing(hitmap1, sigma=self._angle * np.pi / 180)
        h2 = hp.sphtfunc.smoothing(hitmap2, sigma=self._angle * np.pi / 180)

        # Getting rid of the milky way because gmf is too strong + catalog is problematic
        h1[self._mask] = 0
        h2[self._mask] = 0

        # ~6 degrees
        # hm1 = hitmap1.astype(np.float64)
        # hm2 = hitmap2.astype(np.float64)
        # h1 = hp.ud_grade(hm1, 16, power=-2)
        # h2 = hp.ud_grade(hm2, 16, power=-2)
        # print(h1)
        # print(h2)
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

        # return np.dot(h1, h2) / np.linalg.norm(h1) / np.linalg.norm(h2) #np.sqrt(np.dot(h1, h1) * np.dot(h2, h2))
        return np.dot(h1, h2) / np.sum(h1) / np.sum(h2)
        # h1 /= np.sum(h1)
        # h2 /= np.sum(h2)
        # return np.linalg.norm(h1 - h2)
    

class MultipolesTest:
    def __init__(self, ang, exposure):
        self._at = exposure
        self._angle = ang

    def test(self, hitmap):
        # alm = hp.map2alm(hitmap)
        hitmap2 = hp.sphtfunc.smoothing(hitmap, sigma=self._angle * np.pi / 180)
        cl = hp.anafast(hitmap2)
        
        # if   h = h0 (1 + d cos (theta))
        # then c0 = 4pi * h0^2
        # and  c1 = 4pi * h0^2 d^2 / 9 (this 9 is (2l+1)^2)
        # so   d = 3 * sqrt(c1 / c0)
        return np.array([cl[1] / cl[0], cl[2] / cl[0]])

    def other_test(self, hitmap):
        hitmap2 = hp.sphtfunc.smoothing(hitmap / self._at, sigma=self._angle * np.pi / 180)
        # cl = hp.anafast(hitmap2)
        # return np.array([cl[1] / cl[0], cl[2] / cl[0]])

        al = hp.map2alm(hitmap2, lmax=2)
        vec = np.array([np.real(al[3]) * np.sqrt(2), np.imag(al[3]) * np.sqrt(2), np.real(al[1])])
        return vec / np.real(al[0]) * np.sqrt(3)
        


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
    