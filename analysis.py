import numpy as np
import scipy as sp
import healpy as hp


def zoa_mask(nside, mlat):
    def in_zoa(ipix):
        lon, lat = hp.pix2ang(nside, ipix, lonlat=True)
        return (np.abs(lat) < mlat)
    npix = hp.nside2npix(nside)
    return np.where(in_zoa(np.arange(npix)))


class BigMatchedFilterTest:
    def __init__(self, mlat, nside, signal_map=None, signal_vector=None):
        if signal_map is None and signal_vector is None:
            return

        npix = hp.nside2npix(nside)
        
        l, b = hp.pix2ang(nside, np.arange(npix), lonlat=True)
        self._reg0 = np.where(b > 45)
        self._reg1 = np.where((b <= 45) & (l < 180) & (l > 100) & (np.abs(b) > mlat))
        self._reg2 = np.where((b <= 45) & (l <= 100) & (l >= 0) & (np.abs(b) > mlat))
        self._reg3 = np.where((b <= 45) & (l <= 360) & (l >= 270) & (np.abs(b) > mlat))
        self._reg4 = np.where((b <= 45) & (l < 270) & (l >= 180) & (np.abs(b) > mlat))
        self._regions = [self._reg0, self._reg1, self._reg2, self._reg3, self._reg4]

        if signal_map is not None:
            norm_signal = signal_map / np.mean(signal_map)

            self._signal_regions = []
            for region in self._regions:
                self._signal_regions.append(np.sum(norm_signal[region]) / len(region[0]))
            self._signal_regions = np.array(self._signal_regions)
            self._signal_regions = np.log(self._signal_regions)
        else:
            self._signal_regions = np.log(signal_vector)
        

    def test(self, hitmap):
        hitmap_regions = []
        for region in self._regions:
            hitmap_regions.append(np.sum(hitmap[region]))
        hitmap_regions = np.array(hitmap_regions)
        sm = np.sum(hitmap_regions)
        if 0 == sm:
            return 0
        return np.dot(hitmap_regions, self._signal_regions) / sm
    
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
        self._mask = zoa_mask(nside, mlat)
        self._angle = angle
        pass
    
    def test_against(self, hitmap1, hitmap2):
        h1 = hp.sphtfunc.smoothing(hitmap1, sigma=self._angle * np.pi / 180)
        h2 = hp.sphtfunc.smoothing(hitmap2, sigma=self._angle * np.pi / 180)

        # Getting rid of the milky way because gmf is too strong + catalog is problematic
        h1[self._mask] = 0
        h2[self._mask] = 0

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
    def __init__(self, exposure):
        self._at = exposure

        self._gcrot = hp.Rotator(coord=['G', 'C'])
        self._cgrot = hp.Rotator(coord=['C', 'G'])

        mask = self._gcrot.rotate_map_pixel(exposure) > 0
        self._nside = hp.npix2nside(mask.size)
        self._npix = mask.size
        z = hp.pix2vec(self._nside, np.arange(mask.size)[mask])[2]
        zd, zs = 2 * mask.sum() / mask.size, 2 * z.mean()
        cmin, cmax = (np.clip([(zs + zd) / 2, (zs - zd) / 2], -1, 1))

        self._d, self._s, self._p = cmin - cmax, cmin + cmax, cmin * cmax
        self._g = (self._s * self._s - self._p) / 3

        self._u = np.asarray(hp.pix2vec(self._nside, np.arange(self._npix)))


    def dipole_semiexp(self, hitmap):
        normalized_hmap = np.divide(hitmap, self._at, out=np.zeros(self._npix), where=self._at>0)

        I0 = normalized_hmap.sum()
        I = self._u @ normalized_hmap
        I = self._gcrot(I)

        den = self._s * I[2] - 2 * self._g * I0
        f = (self._g-self._p)/(self._g-1)/den
        D = np.array([I[0] * f, I[1] * f, (self._s*I0-2*I[2])/den])
        D = self._cgrot(D)
        return D


class SmallScaleVarTest:
    def __init__(self, angle, nside, mlat):
        self._mask = zoa_mask(nside, mlat)
        self._angle = angle

        npix = hp.nside2npix(nside)
        idx = np.arange(npix)
        l, b = hp.pix2ang(nside, idx, lonlat=True)
        self._reg0 = np.where(b > 45)
        self._reg1 = np.where((b <= 45) & (l < 180) & (l > 100))
        self._reg2 = np.where((b <= 45) & (l <= 100) & (l >= 0))
        self._reg3 = np.where((b <= 45) & (l <= 360) & (l >= 270))
        self._reg4 = np.where((b <= 45) & (l < 270) & (l >= 180))
        self._regs = [self._reg0, self._reg1, self._reg2, self._reg3, self._reg4]

    def test_ent(self, hitmap):
        hmap = hp.sphtfunc.smoothing(hitmap, sigma=self._angle * np.pi / 180)

        hmap[self._mask] = 0

        entr = 0
        for reg in self._regs:
            nmap = hmap[reg]
            nmap /= np.sum(nmap)
            entr += np.sum(sp.special.entr(nmap))
        return np.sum(sp.special.entr(hmap / np.sum(hmap))), entr

