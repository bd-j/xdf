import os, copy
import numpy as np

from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.wcs import WCS

from forcepho.sources import Galaxy, Scene
from forcepho.data import PostageStamp
from forcepho import psf as pointspread
from forcepho.likelihood import lnlike_multi

__all__ = ["cat_to_sourcepars", "prep_scene",
           "make_stamp",
           "Posterior", "Result"] 

base = "/Users/bjohnson/Projects/xdf/data/images/"

mmse_position = (3020, 3470)
mmse_size = (500, 500)


def get_cutout(path, name, position, size):
    sci = fits.open(os.path.join(path, name+"sci.fits"))
    wht = fits.open(os.path.join(path, name+"wht.fits"))
    image = sci[0].data
    image = image.byteswap().newbyteorder()
    weight = wht[0].data
    weight = weight.byteswap().newbyteorder()
    wcs = WCS(sci[0].header)
    cutout_image = Cutout2D(image, position, size, wcs=wcs)
    cutout_weight = Cutout2D(weight, position, size, wcs=wcs)
    image = np.ascontiguousarray(cutout_image.data)
    weight = np.ascontiguousarray(cutout_weight.data)
    rms = np.sqrt(1.0 / weight)
    return image, weight, rms, cutout_image, cutout_image.wcs


def make_xdf_stamp(imroot, psfname, center, size, fwhm=3.0,
                   background=0.0, filtername="H"):

    # Values used to produce the MMSE catalog
    im, wght, rms, cutout, wcs = get_cutout(base, imroot, mmse_position, mmse_size)

    # transpose to get x coordinate on the first axis
    im = im.T
    err = rms.T
    CD = wcs.wcs.cd
    
    # ---- Extract subarray -----
    center = np.array(center)
    # --- here is much mystery ---
    size = np.array(size)
    lo, hi = (center - 0.5 * size).astype(int), (center + 0.5 * size).astype(int)
    xinds = slice(int(lo[0]), int(hi[0]))
    yinds = slice(int(lo[1]), int(hi[1]))
    crpix_stamp = np.floor(0.5 * size)
    crval_stamp = crpix_stamp + lo
    W = np.eye(2)

    # --- MAKE STAMP -------

    # --- Add image and uncertainty data to Stamp ----
    stamp = PostageStamp()
    stamp.pixel_values = im[xinds, yinds] - background
    stamp.ierr = 1./err[xinds, yinds]

    bad = ~np.isfinite(stamp.ierr)
    stamp.pixel_values[bad] = 0.0
    stamp.ierr[bad] = 0.0
    stamp.ierr = stamp.ierr.flatten()

    stamp.nx, stamp.ny = stamp.pixel_values.shape
    stamp.npix = stamp.nx * stamp.ny
    # note the inversion of x and y order in the meshgrid call here
    stamp.ypix, stamp.xpix = np.meshgrid(np.arange(stamp.ny), np.arange(stamp.nx))

    # --- Add WCS info to Stamp ---
    stamp.crpix = crpix_stamp
    stamp.crval = crval_stamp
    stamp.dpix_dsky = np.eye(2)
    stamp.scale = np.linalg.inv(CD * 3600.0) # arcsec per pixel
    stamp.pixcenter_in_full = center
    stamp.lo = lo
    stamp.CD = CD
    stamp.W = W

    # --- Add the PSF ---
    stamp.psf = pointspread.get_psf(psfname, fwhm)

    stamp.filtername = filtername
    return stamp


def prep_scene(sourcepars, filters=["dummy"], splinedata=None, free_sersic=True):

    # --- Get Sources and a Scene -----
    sources = []

    for pars in sourcepars:
        flux, x, y, q, pa, n, rh = copy.deepcopy(pars)
        s = Galaxy(filters=filters, splinedata=splinedata, free_sersic=free_sersic)
        s.sersic = n
        s.rh = rh
        s.flux = flux #np.atleast_1d(flux)
        s.ra = x
        s.dec = y
        s.q = q
        s.pa = np.deg2rad(pa)
        sources.append(s)

    scene = Scene(sources)
    theta = scene.get_all_source_params()
    return scene, theta


def cat_to_sourcepars(catrow):
    ra, dec = catrow["x"], catrow["y"]
    q = np.sqrt(catrow["b"] / catrow["a"])
    pa = 90.0 - catrow["theta"] * 180. / np.pi
    n = 3.0
    S = np.array([[1/q, 0], [0, q]])
    rh = np.mean(np.dot(np.linalg.inv(S), np.array([catrow["a"], catrow["b"]])))
    rh *= 0.06 
    return [ra, dec, q, pa, n, rh]


class Posterior(object):

    def __init__(self, scene, plans, upper=np.inf, lower=-np.inf, verbose=False):
        self.scene = scene
        self.plans = plans
        self._theta = -99
        self.lower = lower
        self.upper = upper
        self.verbose = verbose
        self.ncall = 0

    def evaluate(self, theta):
        Theta = self.complete_theta(theta)
        if self.verbose:
            print(Theta)
            t = time.time()
        ll, ll_grad = lnlike_multi(Theta, scene=self.scene, plans=self.plans)
        lpr, lpr_grad = self.ln_prior_prob(Theta)
        if self.verbose:
            print(time.time() - t)
        self.ncall += 1
        self._lnlike = ll
        self._lnlike_grad = ll_grad
        self._lnprior = lpr
        self._lnprior_grad = lpr_grad
        self._lnp = ll + lpr
        self._lnp_grad = ll_grad + lpr_grad
        self._theta = Theta

    def ln_prior_prob(self, theta):
        return 0.0, np.zeros(len(theta))
        
    def lnprob(self, Theta):
        if np.any(Theta != self._theta):
            self.evaluate(Theta)
        return self._lnp

    def lnprob_grad(self, Theta):
        if np.any(Theta != self._theta):
            self.evaluate(Theta)
        return self._lnp_grad

    def complete_theta(self, theta):
        return theta

    def check_constrained(self, theta):
        """Method that checks parameter values against constraints.  If they
        are above or below the boundaries, the sign of the momentum is flipped
        and theta is adjusted as if the trajectory had bounced off the
        constraint.

        :param theta:
            The parameter vector

        :returns theta:
            the new theta vector

        :returns sign:
            a vector of multiplicative signs for the momenta

        :returns flag:
            A flag for if the values are still out of bounds.
        """

        #initially no flips
        sign = np.ones_like(theta)
        oob = True #pretend we started out-of-bounds to force at least one check
        #print('theta_in ={0}'.format(theta))
        while oob:
            above = theta > self.upper
            theta[above] = 2*self.upper[above] - theta[above]
            sign[above] *= -1
            below = theta < self.lower
            theta[below] = 2*self.lower[below] - theta[below]
            sign[below] *= -1
            oob = np.any(below | above)
            #print('theta_out ={0}'.format(theta))
        return theta, sign, oob


class Result(object):

    def __init__(self):
        self.offsets = None

