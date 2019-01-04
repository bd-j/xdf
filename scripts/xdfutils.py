import os, copy
import numpy as np

from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.wcs import WCS

from forcepho.sources import Galaxy, Scene
from forcepho.data import PostageStamp
from forcepho import psf as pointspread
from forcepho.likelihood import lnlike_multi

try:
    import theano.tensor as tt
except:
    pass


__all__ = ["cat_to_sourcepars", "prep_scene",
           "get_cutout", "make_xdf_stamp",
           "Posterior", "LogLikeWithGrad", "Result"] 

base = "/Users/bjohnson/Projects/xdf/data/images/"

mmse_position = (3020, 3470)
mmse_size = (500, 500)


def get_cutout(path, name, position, size):
    """This is how the image was generated before being run through MMSE
    """
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


def xdf_sky_stamp(imroot, psfname, world, wsize,
                  fwhm=3.0, background=0.0, filtername="H"):
    """Make a stamp with celestial coordinate information.
    """
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    position = SkyCoord(world[0], world[1], unit="deg", frame="icrs")
    size = wsize * u.arcsec

    hdr = fits.getheader(os.path.join(base, imroot+"sci.fits"))
    sci = fits.getdata(os.path.join(base, imroot+"sci.fits"))
    wht = fits.getdata(os.path.join(base, imroot+"wht.fits"))

    wcs = WCS(hdr)
    image = Cutout2D(sci, position, size, wcs=wcs)
    weight = Cutout2D(wht, position, size, wcs=wcs)
    im = np.ascontiguousarray(image.data)
    ivar = np.ascontiguousarray(weight.data)

    # -----------
    # --- MAKE STAMP -------

    # --- Add image and uncertainty data to Stamp, flipping axis order ----
    stamp = PostageStamp()
    stamp.pixel_values = np.array(im).T - background
    stamp.ierr = np.sqrt(ivar).T

    # Clean errors
    bad = ~np.isfinite(stamp.ierr)
    stamp.pixel_values[bad] = 0.0
    stamp.ierr[bad] = 0.0
    stamp.ierr = stamp.ierr.flatten()

    stamp.nx, stamp.ny = stamp.pixel_values.shape
    stamp.npix = stamp.nx * stamp.ny
    # note the inversion of x and y order in the meshgrid call here
    stamp.ypix, stamp.xpix = np.meshgrid(np.arange(stamp.ny), np.arange(stamp.nx))


    # --- Add WCS info to Stamp ---
    psize = np.array(stamp.pixel_values.shape)
    stamp.crpix = np.floor(0.5 * psize)
    stamp.crval = image.wcs.wcs_pix2world(stamp.crpix[None,:], 0)[0, :2]

    CD = image.wcs.wcs.cd
    W = np.eye(2)
    W[0, 0] = np.cos(np.deg2rad(stamp.crval[-1]))
    
    stamp.dpix_dsky = np.matmul(np.linalg.inv(CD), W)
    stamp.scale = np.linalg.inv(CD * 3600.0)
    stamp.CD = CD
    stamp.W = W
    try:
        stamp.wcs = wcs
    except:
        pass

    # --- Add the PSF ---
    stamp.psf = pointspread.get_psf(psfname, fwhm)

    # --- Add extra information ---
    stamp.full_header = dict(hdr)
    if filtername is None:
        stamp.filtername = stamp.full_header["FILTER"]
    else:
        stamp.filtername = filtername

    return stamp


def xdf_cel_stamp(imroot, psfname, world, wsize,
                  fwhm=3.0, background=0.0, filtername="H"):
    """Make a stamp with celestial coordinate information.  A somewhat hackier
    version of xdf_sky_stamp above.
    """

    sci = os.path.join(base, imroot+"sci.fits")
    wht = os.path.join(base, imroot+"wht.fits")

    hdr = fits.getheader(sci)
    data = fits.getdata(sci)
    rms = np.sqrt(1.0 / fits.getdata(wht))

    ast = WCS(hdr, naxis=2)
    CD = ast.wcs.cd
    
    # --- Flip axis order ---
    im = data.T
    err = rms.T

    # ---- Extract subarray -----
    # here we get the center coordinates in pixels (accounting for the transpose above)
    world = np.array(world)
    center = ast.wcs_world2pix(world[None, :], 0)[0, :2]
    # arcsec / pix, assuming square pixels
    plate_scale = np.mean(np.abs(np.linalg.eigvals(3600 * ast.wcs.cd)))
    size = np.array(wsize) / plate_scale

    # --- here is much mystery ---
    lo, hi = (center - 0.5 * size).astype(int), (center + 0.5 * size).astype(int)
    xinds = slice(int(lo[0]), int(hi[0]))
    yinds = slice(int(lo[1]), int(hi[1]))
    # "central" pixel in stamp
    crpix_stamp = np.zeros(2) + np.floor(0.5 * size)
    # pixel coordinates of stamp "center" in full image
    crval_stamp = crpix_stamp + lo
    # world coordinates of stamp "center"
    crval_stamp = ast.wcs_pix2world(crval_stamp[None,:], 0)[0, :2]
    W = np.eye(2)
    W[0, 0] = np.cos(np.deg2rad(crval_stamp[-1]))

    # -----------
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
    stamp.dpix_dsky = np.matmul(np.linalg.inv(CD), W)
    stamp.scale = np.linalg.inv(CD * 3600.0)
    stamp.pixcenter_in_full = center
    stamp.lo = lo
    stamp.CD = CD
    stamp.W = W
    try:
        stamp.wcs = ast
    except:
        pass

    # --- Add the PSF ---
    stamp.psf = pointspread.get_psf(psfname, fwhm)

    # --- Add extra information ---
    stamp.full_header = dict(hdr)
    if filtername is None:
        stamp.filtername = stamp.full_header["FILTER"]
    else:
        stamp.filtername = filtername

    return stamp

    
    
def xdf_pixel_stamp(imroot, psfname, center, size, fwhm=3.0,
                    background=0.0, filtername="H"):
    """Make a stamp with pixel coordinate information
    """
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

    def __init__(self, scene, plans, lnlike=lnlike_multi, lnprior=None,
                 transform=None, upper=np.inf, lower=-np.inf, verbose=False):
        self.scene = scene
        self.plans = plans
        self.lnlike = lnlike
        if lnprior is not None:
            self.lnprior = lnprior
        self.T = transform
        self.ncall = 0
        self._theta = -99
        self._z = -99
        self.lower = lower
        self.upper = upper

    def evaluate(self, z):
        """
        :param z:
            The untransformed (sampling) parameters which have a prior
            distribution attached.

        Theta are the transformed forcepho native parameters.  In the default
        case these are these are the same as thetaprime, i.e. the
        transformation is the identity.
        """
        Theta = self.transform(z)
        ll, ll_grad = self.lnlike(Theta, scene=self.scene, plans=self.plans)
        lpr, lpr_grad = self.lnprior(Theta)
       
        self.ncall += 1
        self._lnlike = ll
        self._lnlike_grad = ll_grad
        self._lnprior = lpr
        self._lnprior_grad = lpr_grad

        self._lnp = ll + lpr + self._lndetjac
        self._lnp_grad = (ll_grad + lpr_grad) * self._jacobian + self._lndetjac_grad
        self._theta = Theta
        self._z = z

    def lnprior(self, Theta):
        return 0.0, 0.0
        
    def lnprob(self, z):
        if np.any(z != self._z):
            self.evaluate(z)
        return self._lnp

    def lnprob_grad(self, z):
        if np.any(z != self._z):
            self.evaluate(z)
        return self._lnp_grad

    def transform(self, z):
        if self.T is not None:
            self._jacobian = self.T.jacobian(z)
            self._lndetjac = self.T.lndetjac(z)
            self._lndetjac_grad = self.T.lndetjac_grad(z)
            return self.T.transform(z)
        else:
            self._jacobian = 1.
            self._lndetjac = 0
            self._lndetjac_grad = 0
            return np.array(z)

    def check_constrained(self, theta):
        """Method that checks parameter values against constraints.  If they
        are above or below the boundaries, the sign of the momentum is flipped
        and theta is adjusted as if the trajectory had bounced off the
        constraint.  This is only useful for bd-j/hmc backends.

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


class ModelGradOp(tt.Op):
    """Wraps the Posterior object lnprob_grad() method in a theano tensor
    operation
    """

    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, model):
        self.model = model

    def perform(self, node, inputs, outputs):
        z, = inputs
        ll_grads = self.model.lnprob_grad(z)
        outputs[0][0] = ll_grads


class LogLikeWithGrad(tt.Op):
    """Wraps the Posterior object lnprob() and lnprob_grad() methods in theano
    tensor operations
    """
    
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, model):
        self.model = model
        self.GradObj = ModelGradOp(self.model)

    def perform(self, node, inputs, outputs):
        z, = inputs
        logl = self.model.lnprob(z)
        outputs[0][0] = np.array(logl)

    def grad(self, inputs, g):
        z, = inputs
        return [g[0] * self.GradObj(z)]
    

class Result(object):

    def __init__(self):
        self.offsets = None

