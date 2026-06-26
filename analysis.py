import numpy as np
import scipy as sp
import healpy as hp
import matplotlib.pyplot as plt


def zoa_mask(nside):
    def in_zoa(ipix):
        lon, lat = hp.pix2ang(nside, ipix, lonlat=True)
        return (np.abs(lat) < 5) | ((np.abs(lat) < 10) & (np.abs((lon + 180) % 360 - 180) < 30))
    npix = hp.nside2npix(nside)
    return np.where(in_zoa(np.arange(npix)))

def zoa_mask2(nside):
    def in_zoa(ipix):
        lon, lat = hp.pix2ang(nside, ipix, lonlat=True)
        return (np.abs(lat) < 20)
    npix = hp.nside2npix(nside)
    return np.where(in_zoa(np.arange(npix)))

def zoa_maskc(nside, mlat):
    def in_zoa(ipix):
        lon, lat = hp.pix2ang(nside, ipix, lonlat=True)
        return (np.abs(lat) < mlat)
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


class BigMatchedFilterTest:
    def __init__(self, mlat, signal=None):
        if signal is None:
            return

        nside = hp.get_nside(signal)
        npix = len(signal)
        
        l, b = hp.pix2ang(nside, np.arange(npix), lonlat=True)
        self._reg0 = np.where(b > 45)
        self._reg1 = np.where((b <= 45) & (l < 180) & (l > 100) & (np.abs(b) > mlat))
        self._reg2 = np.where((b <= 45) & (l <= 100) & (l >= 0) & (np.abs(b) > mlat))
        self._reg3 = np.where((b <= 45) & (l <= 360) & (l >= 270) & (np.abs(b) > mlat))
        self._reg4 = np.where((b <= 45) & (l < 270) & (l >= 180) & (np.abs(b) > mlat))
        self._regions = [self._reg0, self._reg1, self._reg2, self._reg3, self._reg4]


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
    
    def visualize(self, hitmap):
        tmap = np.zeros(len(hitmap))
        hmap = np.zeros(len(hitmap))
        for region, s in zip(self._regions, self._signal_regions):
            hmap[region] = np.sum(hitmap[region]) / len(region)
            tmap[region] = np.exp(s) / len(region)

        hp.mollview(hmap, title="")
        hp.mollview(tmap, title="")

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
    

class BigMatchedFilterTest2:
    def __init__(self, signal=None):
        if signal is None:
            return

        nside = hp.get_nside(signal)
        npix = len(signal)
        
        self._regions = [[], [], [], [], [], []]
        
        for ipix in range(npix):
            lon, lat = hp.pix2ang(nside, ipix, lonlat=True)

            if (0 <= lon < 120):
                self._regions[0].append(ipix)
            if (120 <= lon < 150):
                self._regions[1].append(ipix)
            if (150 <= lon < 180):
                self._regions[2].append(ipix)
            if (180 <= lon < 210):
                self._regions[3].append(ipix)
            if (210 <= lon < 240):
                self._regions[4].append(ipix)
            if (240 <= lon < 360):
                self._regions[5].append(ipix)
            

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
    
    def visualize(self, hitmap):
        tmap = np.zeros(len(hitmap))
        hmap = np.zeros(len(hitmap))
        for region, s in zip(self._regions, self._signal_regions):
            hmap[region] = np.sum(hitmap[region]) / len(region)
            tmap[region] = np.exp(s) / len(region)

        hp.mollview(hmap, title="")
        hp.mollview(tmap, title="")

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
    def __init__(self, angle, mlat, nside):
        self._mask = zoa_maskc(nside, mlat)
        self._angle = angle
        pass
    
    def test_against(self, hitmap1, hitmap2):
#         h1 = self._dmap @ hitmap1
        # h2 = self._dmap @ hitmap2
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

        proj = 0
        s1 = 0
        s2 = 0
        for i in range(len(h1)):
            proj += h1[i] * h2[i]
            s1 += h1[i]
            s2 += h2[i]
        return proj / s1 / s2

        # return np.dot(h1, h2) / np.linalg.norm(h1) / np.linalg.norm(h2) #np.sqrt(np.dot(h1, h1) * np.dot(h2, h2))
        return np.dot(h1, h2) / np.sum(h1) / np.sum(h2)
        # h1 /= np.sum(h1)
        # h2 /= np.sum(h2)
        # return np.linalg.norm(h1 - h2)
    

class MultipolesTest:
    def __init__(self, angle, exposure):
        self._at = exposure
        self._angle = angle

    def test(self, hitmap):
        # alm = hp.map2alm(hitmap)
        hitmap2 = self._dmap @ hitmap
        cl = hp.anafast(hitmap2)
        
        # if   h = h0 (1 + d cos (theta))
        # then c0 = 4pi * h0^2
        # and  c1 = 4pi * h0^2 d^2 / 9 (this 9 is (2l+1)^2)
        # so   d = 3 * sqrt(c1 / c0)
        return np.array([cl[1] / cl[0], cl[2] / cl[0]])

    def other_test(self, hitmap):
        hitmap2 = hp.sphtfunc.smoothing(hitmap, sigma=self._angle * np.pi / 180) / self._at
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
    

def ad(map, compmap):
    n = len(map)
    map.sort()
    s = 0
    j = 0
    for i in range(n):
        while (j < n) and (compmap[j] < map[i]):
            j += 1
        s += (2*i+1)/n*np.log((j+.2)/n) + (2*(n-1-i)+1)/n*np.log(1-(j-.2)/n)
    return -n-s

class SmallScaleVarTest:
    def __init__(self, angle, nside, mlat):
        # self._mask = zoa_mask(nside)
        # self._mask = zoa_mask2(nside)
        self._mask = zoa_maskc(nside, mlat)
        self._angle = angle

        npix = hp.nside2npix(nside)
        idx = np.arange(npix)
        cgrot = hp.Rotator(coord=['G', 'C'])
        idec = cgrot(hp.pix2ang(nside, idx))[0]
        self._northmap = np.where(idec <= np.pi/2)
        self._southmap = np.where(idec > np.pi/2)
        l, b = hp.pix2ang(nside, idx, lonlat=True)
        self._reg0 = np.where(b > 45)
        self._reg1 = np.where((b <= 45) & (l < 180) & (l > 100))
        self._reg2 = np.where((b <= 45) & (l <= 100) & (l >= 0))
        self._reg3 = np.where((b <= 45) & (l <= 360) & (l >= 270))
        self._reg4 = np.where((b <= 45) & (l < 270) & (l >= 180))
        self._regs = [self._reg0, self._reg1, self._reg2, self._reg3, self._reg4]

    def test(self, hitmap):
        hmap = self._dmap @ hitmap

        hmap = np.delete(hmap, self._mask)

        hmap /= np.sum(hmap)

        nn = sp.stats.kstest(hmap, self._n).statistic
        pp = sp.stats.kstest(hmap, self._p).statistic
        return nn, pp
    
    def test2(self, hitmap, c='black'):
        #hmap = self._dmap @ hitmap
        hmap = hp.sphtfunc.smoothing(hitmap, sigma=16.5 * np.pi / 180)

        hmap = np.delete(hmap, self._mask)

        hmap /= np.sum(hmap)

        nn = ad(hmap, self._n)
        pp = ad(hmap, self._p)
        return nn, pp

        hist, be = np.histogram(hmap, bins=np.linspace(0, max(hmap)))
        cist = np.cumsum(hist) / np.sum(hist)
        plt.plot(be[:-1] / np.sum(hmap), cist, color=c, alpha=.3)
        # TOPHAT

    def test_ent(self, hitmap):
        hmap = hp.sphtfunc.smoothing(hitmap, sigma=self._angle * np.pi / 180)

        #hmap = np.delete(hmap, self._mask)
        hmap[self._mask] = 0

        entr = 0
        for reg in self._regs:
            nmap = hmap[reg]
            nmap /= np.sum(nmap)
            entr += np.sum(sp.special.entr(nmap))
        return np.sum(sp.special.entr(hmap / np.sum(hmap))), entr

