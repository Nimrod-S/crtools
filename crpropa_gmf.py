import colorsys
import tqdm
import numpy as np
import scipy as sp
import healpy as hp
import matplotlib.pyplot as plt

from crpropa import *

from cosmology import *

# Z for each A
# A=5 and A=8 are made up, A=1 is just hydrogen
STABLE_ISOTOPES = {2: 1, 3: 2, 4: 2, 6: 3, 7: 4, 9: 4, 10: 5, 11: 5, 12: 6, 13: 6, 14: 7, 15: 7, 16: 8, 17: 8, 18: 8, 19: 9, 20: 10, 21: 10, 22: 10, 23: 11, 24: 12, 25: 12, 26: 12, 27: 13, 28: 14, 29: 14, 30: 14, 31: 15, 32: 16, 33: 16, 34: 16, 36: 18, 35: 17, 37: 18, 38: 18, 40: 20, 39: 19, 41: 20, 42: 20, 43: 20, 44: 22, 45: 21, 46: 22, 47: 22, 48: 22, 49: 23, 50: 24, 51: 24, 52: 24, 53: 25, 54: 26, 55: 26, 56: 26, 1: 1, 5: 3, 8: 4}


def define_simulation(modelname, modeltype=0, dist=30):
    simulation = ModuleList()

    if modelname == "UF23":
        # model variations (see Tab.2 of UF23 paper)
        # enum ModelType {
        #     base,
        #     neCL,
        #     expX,
        #     spur,
        #     cre10,
        #     synCG,
        #     twistX,
        #     nebCor
        # };
        gmf = UF23FieldWrap(modeltype)
    elif modelname == "JF12":
        gmf = JF12Field()
    elif modelname == "KST24":
        gmf = KST24FieldWrap()
    
    simulation.add(PropagationCK(gmf, 1e-4, 0.1 * parsec, 100 * parsec)) # TODO understand these numbers


    observer = Observer()
    observer.add(ObserverSurface(Sphere(Vector3d(0), dist * kpc)))
    # observer.add(observerSurface())

    # simulation.add(observer)
    # simulation.add(SphericalBoundary(Vector3d(0, 0, 0), dist * kpc))
    simulation.add(CylindricalBoundary(Vector3d(0, 0, 0), 2 * kpc, 20 * kpc))
    # Propagation type

    return simulation


def define_ray(r, ang):

    pid = -nucleusId(1, 1) # antiproton
    energy = r * eV # For proton, rigidity is just the energy
    startpos = Vector3d(-8.5, 0, 0) * kpc # Solar system in MW

    v = hp.ang2vec(*ang)
    direction = Vector3d(v[0], v[1], v[2])

    p = ParticleState(pid, energy, startpos, direction)
    return Candidate(p)


def backtrace(sim, rays):
    # If many
    for r in rays:
        sim.run(r)
        d = r.current.getDirection()
        yield hp.vec2ang(np.array([d.x, d.y, d.z]))
    
    return

def backtracev(sim, rays, nside):
    sim.run(rays)
    ds = [ray.current.getDirection() for ray in rays]
    trajs = [ray.getTrajectoryLength() / kpc for ray in rays]
    return [hp.vec2pix(nside, d.x, d.y, d.z) for d in ds], trajs
    return [hp.vec2ang(np.array([d.x, d.y, d.z])) for d in ds]

def proplens(simname, rig, nside, modeltype=0):
    # Returns lens, so lens[pixel on earth]=pixel outside galaxy
    npix = hp.nside2npix(nside)
    angs0 = [hp.pix2ang(nside, ipix) for ipix in np.arange(npix)]
    
    rays = CandidateVector()
    for ang in angs0:
        rays.push_back(CandidateRefPtr(define_ray(rig, ang)))

    sim = define_simulation(simname, modeltype=modeltype)
    
    return backtracev(sim, rays, nside)[0]

def proplength(simname, rig, nside, modeltype=0):
    # Returns lens, so lens[pixel on earth]=pixel outside galaxy
    npix = hp.nside2npix(nside)
    angs0 = [hp.pix2ang(nside, ipix) for ipix in np.arange(npix)]
    
    rays = CandidateVector()
    for ang in angs0:
        rays.push_back(CandidateRefPtr(define_ray(rig, ang)))

    sim = define_simulation(simname, modeltype=modeltype)
    
    return backtracev(sim, rays, nside)[1]

def ang2col(ang):
    th, ph = ang
    hue = ph / 2 / np.pi
    light = th / np.pi
    sat = 1
    rgb = colorsys.hls_to_rgb(hue, light, sat)
    rgb = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

def region(nside, name):
    npix = hp.nside2npix(nside)
    angs0 = [hp.pix2ang(nside, ipix) for ipix in np.arange(npix)]
    
    if name == "ALL":
        return angs0
    if name == "1":
        angs0 = [ang for ang in angs0 if (180 > ang[1]*180/np.pi > 150)]
    elif name == "2":
        angs0 = [ang for ang in angs0 if (210 > ang[1]*180/np.pi > 180)]
    elif name == "cent":
        angs0 = [hp.pix2ang(nside, p) for p in hp.query_disc(nside, hp.ang2vec(-43, 28, lonlat=True), 1)]

    return angs0


def fullskycolor(nside, regionname):
    npix = hp.nside2npix(nside)
    z = np.zeros(npix)

    angs0 = region(nside, regionname)

    colors = [ang2col(ang) for ang in angs0]

    ths0 = []
    phs0 = []
    for ang in angs0: th, ph = ang; ths0.append(th); phs0.append(ph)
    hp.mollview(z, cmap="binary")
    hp.projscatter(np.array(ths0).ravel(), np.array(phs0).ravel(), c=colors)
    print("HELLO")

    for model in ["JF12", "KST24", "UF23"]:
        s = define_simulation(model)
        rays = [define_ray(5e18, ang) for ang in angs0]

        ths = []
        phs = []
        cols2 = []
        for ang in backtrace(s, rays):
            th, ph = ang
            ths.append(th)
            phs.append(ph)
            cols2.append(ang2col(ang))

        hp.mollview(z, cmap="binary")

        hp.projscatter(np.array(ths).ravel(), np.array(phs).ravel(), c=colors)
        
        hp.mollview(z, cmap="binary")
        hp.projscatter(np.array(ths0).ravel(), np.array(phs0).ravel(), c=cols2)

        plt.show()




def main():
    # s = define_simulation("KST24")
    # r = define_ray(5e18, (np.pi/3, np.pi/2))
    # a = list(backtrace(s, [r]))[0]
    # print(a)

    # fullskycolor(64, "ALL")
    # fullskycolor(64, "1")
    # fullskycolor(64, "2")

    return



if "__main__" == __name__:
    main()
