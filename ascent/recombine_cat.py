#!/usr/bin/env python

"""
A script to get the fluxes (and other properties) of many objects and compare to the 3dhst values

"""

import sys, os
from os.path import join as pjoin
import numpy as np
import h5py
import matplotlib.pyplot as pl
pl.ion()

from forcepho.fitting import Result
from forcepho.likelihood import lnlike_multi, make_image, WorkPlan
from patch_conversion import patch_conversion, zerocoords, set_inactive

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits

#_print = print
#print = lambda *args,**kwargs: _print(*args,**kwargs, file=sys.stderr, flush=True)

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

path_to_data = "/Users/bjohnson/Projects/xdf/data/"
path_to_results = "/Users/bjohnson/Projects/xdf/results/"
splinedata = pjoin(path_to_data, "sersic_mog_model.smooth=0.0150.h5")
psfpath = pjoin(path_to_data, "psfs", "mixtures")

threedhst_cat = fits.getdata("/Users/bjohnson/Projects/xdf/data/catalogs/goodss_3dhst.v4.1.cat.FITS")
threed_extras = ["ra", "dec", "tot_cor", "a_image", "b_image", "theta_J2000", "kron_radius"]
cols = [(b, np.float) for b in bands]
cols += [(err.format(b), np.float) for b in bands]
cols += [(n, np.float) for n in threed_extras]
cols += [("id", np.int)]
threed_dtype = np.dtype(cols)


def query_3dhst(ra, dec, bands):
    hst = SkyCoord(ra=threedhst_cat['ra'], dec=threedhst_cat['dec'], unit=(u.deg, u.deg))
    force = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
    idx, sep, _ = force.match_to_catalog_sky(hst)

    row = np.zeros(1, dtype=threed_dtype)
    catrow = threedhst_cat[idx]
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


def add_source(source, chain):

    row = np.zeros(1, dtype=catalog_dtype)
    for band in source.filternames:
        ind = source.filter_index(band)
        f = chain[:, ind]
        row[band] = f.mean()
        row[err.format(band)] = f.std()
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


def get_patch(resultname, splinedata=splinedata, psfpath=psfpath):
    """
    """
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
    stamps, miniscene = patch_conversion(patchname, splinedata, psfpath, nradii=9)
    miniscene = set_inactive(miniscene, [stamps[0], stamps[-1]], nmax=nsources)
    zerocoords(stamps, miniscene, sky_zero=np.array([pra, pdec]))
    
    return stamps, miniscene, chain, pra, pdec



if __name__ == "__main__":
    pids = [90, 91, 157, 159, 183, 274, 382, 653]
    pid = 159
    force, threed = [], []
    for pid in pids:
        fp, td = add_patch(pid)
        force.append(fp)
        threed.append(td)
        
    force = np.concatenate(force)
    threed = np.concatenate(threed)
    
    fits.writeto("ascent_xdf_forcepho.fits", force, overwrite=True)
    fits.writeto("ascent_xdf_threedhst.fits", threed, overwrite=True)
    
    good = force["patchID"] != 274
    force = force[good]
    threed = threed[good]
    
    xx = np.linspace(-3, 3, 100)
    
    # --- Plot magnitudes ---
    mf, mt = -2.5*np.log10(force["f160w"]), -2.5*np.log10(threed["f160w"])
    mf_e, mt_e = 1.086*(force["f160w_err"]/ force["f160w"]), 1.086*(threed["f160w_err"]/ threed["f160w"])
    ffig, faxes = pl.subplots(1, 2, figsize=(20, 8))
    fax = faxes[0]
    fax.errorbar(mt, mf, xerr=mt_e, yerr=mf_e, marker="o", linestyle="", color="slateblue")
    fax.set_xlabel(r"$m_{f160w} - 26$ (3DHST)")
    fax.set_ylabel(r"$m_{f160w} - 26$ (force)")
    fax.plot(xx, xx, linestyle = "--", color="red")
    fax.plot(xx, xx-0.2, linestyle=":", color="red", label=r"$\pm 0.2 \, mag$")
    fax.plot(xx, xx+0.2, linestyle=":", color="red")
    fax.legend()
    
    fax = faxes[1]
    chi = (mf - mt) / np.sqrt(mf_e**2 + mt_e**2)
    fax.hist(chi, bins=20, alpha=0.5, color="slateblue")
    fax.set_xlabel(r"$\chi = \frac{\Delta m}{\sqrt{\sigma_{force}^2 + \sigma_{3D}^2}}$")   
    fax.text(0.1, 0.7, "std. dev. = {:2.2f}".format(chi.std()), transform=fax.transAxes)
    ffig.savefig("flux_comparison.pdf")
    
    zz = zip(force["patchID"], force["sourceID"], mt, chi)
    
    _ = [print(z) for z in zz]
    
    # --- Plot colors ---
     