import os, copy
import numpy as np

from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.wcs import WCS

from forcepho.sources import Galaxy, Scene
from forcepho.data import PostageStamp
from forcepho import psf as pointspread

try:
    import theano.tensor as tt
except:
    pass


__all__ = ["setup_patch", "cat_to_sourcepars", "prep_scene",
           "convert_region", "get_cutout", "get_mmse_60mas_wcs",
           "xdf_pixel_stamp", "xdf_sky_stamp",
           "Result"]

base = "/Users/bjohnson/Projects/xdf/data/images/"

mmse_position = (3020, 3470)
mmse_size = (500, 500)

psfpaths = {"f814w": "../data/psfs/mixtures/gmpsf_30mas_hst_f814w_ng4.h5",
            "f160w": "../data/psfs/mixtures/gmpsf_hst_f160w_ng3.h5"
            }
imnames = {"f814w": "hlsp_xdf_hst_acswfc-30mas_hudf_f814w_v1_",
           "f160w": "hlsp_xdf_hst_wfc3ir-60mas_hudf_f160w_v1_"
           }


def setup_xdf_patch(args, filters=[], sky=True, single_source=True, mmse_cat=None):

    # Get region and choose sources.
    # Based on either pixel coordinates or celestial coordinates

    cat = mmse_cat
    pixel_region = args.ra <= 0

    if pixel_region:
        xlo, xhi, ylo, yhi = tuple(args.corners)
        lo, hi = np.array([xlo, ylo]), np.array([xhi, yhi])
        pcenter = np.round((hi + lo)  / 2)
        psize = hi - lo
        scenter, ssize = convert_region(pcenter, psize, direction="to sky")
        string = "x{:.0f}_y{:.0f}_{}".format(pcenter[0], pcenter[1], "".join(filters))

        sel = ((cat["x"] > lo[0]) & (cat["x"] < hi[0]) &
               (cat["y"] > lo[1]) & (cat["y"] < hi[1]))

    else:
        scenter = np.array([args.ra, args.dec])
        ssize = args.size
        pcenter, psize = convert_region(scenter, ssize, direction="to pixels")
        string = "ra{:.4f}_dec{:.4f}_{}".format(args.ra, args.dec, "".join(filters))

        from astropy import units as u
        from astropy.coordinates import SkyCoord
        coords = SkyCoord(cat["ra"] * u.degree, cat["dec"] * u.degree, frame="icrs")
        sep = coords.separation(SkyCoord(args.ra * u.deg, args.dec * u.deg, frame="icrs"))
        if single_source:
            sel = [np.argmin(sep)]
        else:
            # FIXME: Choose sources based on whether they actually appear in any
            # stamp (+ buffer)
            sel = sep < (np.sqrt(2) * np.mean(size) * u.arcsec)

    if sky:
        stamper = xdf_sky_stamp
        celestial = True
        center, size = scenter, ssize
    else:
        stamper = xdf_pixel_stamp
        celestial = False
        center, size = pcenter, psize

    # --- Set up Scene ---
    fluxes = [[f*2.0 for filt in filters] for f in cat[sel]["flux"]]
    sourcepars = [tuple([flux] + cat_to_sourcepars(s, celestial=celestial))
                  for flux, s in zip(fluxes, cat[sel])]

    # --- Make stamps ---
    stamps = [stamper(imnames[f], psfpaths[f], center, size, filtername=f)
              for f in filters]

    return sourcepars, stamps, string


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


def cat_to_sourcepars(catrow, celestial=False):
    if celestial:
        ra, dec = catrow["ra"], catrow["dec"]
    else:
        ra, dec = catrow["x"], catrow["y"]
    q = np.sqrt(catrow["b"] / catrow["a"])
    pa = 90.0 - catrow["theta"] * 180. / np.pi
    n = 3.0
    S = np.array([[1/q, 0], [0, q]])
    rh = np.mean(np.dot(np.linalg.inv(S), np.array([catrow["a"], catrow["b"]])))
    rh *= 0.06  # Handle mmse 0.06 arcsec pixels
    return [ra, dec, q, pa, n, rh]


def convert_region(center, size, direction="to pixels"):
    wcs = get_mmse_60mas_wcs()
    CD = wcs.wcs.cd * 3600 # convert to arcsec
    size = np.zeros(2) + np.array(size)
    if direction == "to pixels":
        pcenter = wcs.all_world2pix(center[0], center[1], 0)
        psize = np.dot(np.linalg.inv(CD), size)
        return pcenter, np.abs(psize)
    elif direction == "to sky":
        scenter = wcs.all_pix2world(center[0], center[1], 0)
        ssize = np.dot(CD, size)
        return scenter, np.abs(ssize)
    else:
        raise(ValueError)


def get_mmse_60mas_wcs():
    sci = fits.open("../data/images/hlsp_xdf_hst_wfc3ir-60mas_hudf_f160w_v1_sci.fits")
    image = sci[0].data
    wcs = WCS(sci[0].header)
    cutout_image = Cutout2D(image, mmse_position, mmse_size, wcs=wcs)
    return cutout_image.wcs


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
    # note size should be (ny, nx)
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

    
def xdf_pixel_stamp(imroot, psfname, center, size, fwhm=3.0,
                    background=0.0, filtername="H"):
    """Make a stamp with pixel coordinate information.
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

    

class Result(object):

    def __init__(self):
        self.offsets = None

