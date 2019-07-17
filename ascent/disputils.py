#!/usr/bin/env python

import sys, os
from os.path import join as pjoin

import numpy as np
import h5py

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

# for each band get the 3DHST flux column name, and the XDF zeropoint
band_map = {"f160w": ("f_F160W", 25.94),
            "f140w": ("f_F140W", 26.45),
            "f125w": ("f_F125W", 26.23),
            "f105w": ("f_F105W", 26.27),
            "f435w": ("f_F435W", 25.68),
            "f606w": ("f_F606W", 26.51),
            "f775w": ("f_F775W", 25.69),
            "f814w": ("f_F814Wcand", 25.94),
            "f850lp": ("f_F850LP", 24.87),
            }

err = "{}_err"
shape_pars = ["ra", "dec", "q", "pa", "sersic", "rh"]
bands = list(band_map.keys())
bands.sort()

cols = [(b, np.float) for b in bands]
cols += [(err.format(b), np.float) for b in bands]
cols += [(p, np.float) for p in shape_pars]
cols += [(err.format(p), np.float) for p in shape_pars]
cols += [("patchID", np.int), ("sourceID", np.int)]
catalog_dtype = np.dtype(cols)

threedcatname = "/Users/bjohnson/Projects/xdf/data/catalogs/goodss_3dhst.v4.1.cat.FITS"
if os.path.exists(threedcatname):
    threedhst_cat = fits.getdata(threedcatname)

threed_extras = ["ra", "dec", "tot_cor", "a_image", "b_image", "theta_J2000", "kron_radius"]
cols = [(b, np.float) for b in bands]
cols += [(err.format(b), np.float) for b in bands]
cols += [(n, np.float) for n in threed_extras]
cols += [("id", np.int), ("sep", np.float)]
threed_dtype = np.dtype(cols)


def query_3dhst(ra, dec, bands):
    """Return a row of the `threedhst_cat` array with coordinates closest to 
    the supplied ra and dec.  Convert fluxes to XDF counts/sec and populate the
    "sep" field of the output.
    
    :returns sep:
        Separation between supplied ra, dec and 3DHST ra, dec, in arcsec.
    
    :returns row:
        A one-element structured array (i.e. a table row) with 3DHST information.

    """
    hst = SkyCoord(ra=threedhst_cat['ra'], dec=threedhst_cat['dec'], unit=(u.deg, u.deg))
    force = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
    idx, sep, _ = force.match_to_catalog_sky(hst)

    row = np.zeros(1, dtype=threed_dtype)
    catrow = threedhst_cat[idx]
    row["sep"] = sep.to("arcsec").value
    for n in threed_extras:
        row[0][n] = catrow[n]
    for b in bands:
        try:
            # account for differnce between 3DHST (25) and XDF zeropoints
            tband, zp = band_map[b]
            conv = 10**((zp - 25)/2.5)
            row[0][b] = catrow[tband] * conv
            row[0]["{}_err".format(b)] = catrow[tband.replace("f_", "e_")] * conv
            row[0]["id"] = catrow["id"]
        except(KeyError):
            pass

    return sep, row


def get_color(cat, bands, err=err):
    """Get a color (in AB mags) and error theron.
    
    :param cat:
        structured ndarray giving the catalog.
        
    :param bands: 2-tuple
        2 strings giving the names of the bands composing the color.  e.g.
        ("f814w", "f160w") will yield the f814w-f160w color and uncertainty in
        AB mags
        
    :returns color:
        The color, same length as `cat`
        
    :returns unc:
        The uncertainty on the color, same length as `cat`
    """
    color = -2.5*np.log10(cat[bands[0]] / cat[bands[1]])
    merr = [1.086 * cat[err.format(b)] / cat[b] for b in bands]
    cerr = np.hypot(*merr)
    return color, cerr


def get_mag(cat, band, err=err):
    """Get AB magnitude and uncertainty in the requested band

    :param cat:
        structured array of catalog values

    :param band:
        string, e.g. "f160w", specifying the band.

    :param err: (optional)
        format string for the name of the error column. 
    """
    mag = band_map[band][1] - 2.5*np.log10(cat[band])
    mag_e = 1.086*(cat[err.format(band)]/ cat[band])
    return mag, mag_e


def check_ivar(ra, dec, band):
    """Return the XDF inverse-variance (proportional to exposure time) in the 
    requested band at a given location on the sky
    """
    imname = f"../data/images/hlsp_xdf_hst_wfc3ir-60mas_hudf_{band}_v1_wht.fits"
    wht = fits.getdata(imname).T
    wcs = WCS(fits.getheader(imname))
    x, y = wcs.all_world2pix(ra, dec, 0, ra_dec_order=True)
    return wht[x.astype(int), y.astype(int)]


def get_patch(resultname):
    """
    """
    
    path_to_data = "/Users/bjohnson/Projects/xdf/data/"
    path_to_results = "/Users/bjohnson/Projects/xdf/results/"
    splinedata = pjoin(path_to_data, "sersic_mog_model.smooth=0.0150.h5")
    psfpath = pjoin(path_to_data, "psfs", "mixtures")

    
    # Get the results
    with h5py.File(resultname, "r") as f:
        chain = f["chain"][:]
        patch = f.attrs["patchname"]
        pra, pdec = f.attrs["sky_reference"]
        ncall = f.attrs["ncall"]
        twall = f.attrs["wall_time"]
        lower = f.attrs["lower_bounds"]
        upper = f.attrs["upper_bounds"]
        nsources = f.attrs["nactive"]

    patchname = pjoin(path_to_data, "patches", "20190612_9x9_threedhst", 
                      os.path.basename(patch))

    # --- Prepare Patch Data ---
    from patch_conversion import patch_conversion, zerocoords, set_inactive
    stamps, miniscene = patch_conversion(patchname, splinedata, psfpath, nradii=9)
    miniscene = set_inactive(miniscene, [stamps[0], stamps[-1]], nmax=nsources)
    zerocoords(stamps, miniscene, sky_zero=np.array([pra, pdec]))
    
    return stamps, miniscene, chain, pra, pdec



def add_source(source, chain, covar=False):

    row = np.zeros(1, dtype=catalog_dtype)
    for band in source.filternames:
        ind = source.filter_index(band)
        f = chain[:, ind]
        if covar:
            C = np.cov(chain.T)
            f160w = chain[:, source.filter_index("f160w")]
        else:
            f160w = 1.0
        row[band] = f.mean()
        row[err.format(band)] = (f/f160w).std()
    start = source.nband
    for i, p in enumerate(shape_pars):
        ind = start + i
        row[p] = chain[:, ind].mean()
        row[err.format(p)] = chain[:, ind].std()
    
    return row


def add_patch(pid):
    resultname = "results/max10-patch_udf_withcat_{}_result.h5".format(pid)
    stamps, scene, chain, pra, pdec = get_patch(resultname)

    rows, threed = [], []
    for isource in range(len(scene)):
        source = scene.sources[isource]
        ra, dec = source.ra + pra, source.dec + pdec
        sep, tdrow = query_3dhst(ra, dec, source.filternames)
        start = int(np.sum([s.nparam for s in scene.sources[:isource]]))
        stop = start + int(source.nparam)

        row = add_source(source, chain[:, start:stop])
        row["ra"] += pra
        row["dec"] += pdec
        row["patchID"] = pid
        row["sourceID"] = isource
        rows.append(row)
        threed.append(tdrow)

    return np.concatenate(rows), np.concatenate(threed)
