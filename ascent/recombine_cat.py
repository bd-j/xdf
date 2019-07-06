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

from astropy.io import fits
from disputils import get_patch, query_3dhst, get_color, get_mag, check_ivar


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



  
    
if __name__ == "__main__":
    
    if True:
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
        #sys.exit()
    else:
        force = fits.getdata("ascent_xdf_forcepho.fits")
        threed = fits.getdata("ascent_xdf_threedhst.fits")


    good = force["patchID"] != 274
    force = force[good]
    threed = threed[good]
    
    xx = np.linspace(-3, 3, 100) + 26.0
    
    # --- Plot magnitudes ---
    ref_band = "f160w"
    mf, mf_e = get_mag(force, ref_band)
    mt, mt_e = get_mag(threed, ref_band)
    ivar = check_ivar(force["ra"], force["dec"], ref_band)
    candels_depth = np.log10(ivar) < 5
    sel = ~candels_depth

    ffig, faxes = pl.subplots(1, 2, figsize=(10, 4))
    fax = faxes[0]
    fax.errorbar(mt[~sel], mf[~sel], xerr=mt_e[~sel], yerr=mf_e[~sel], marker="o", linestyle="", 
                 linewidth=2, color="slateblue")
    fax.errorbar(mt[sel], mf[sel], xerr=mt_e[sel], yerr=mf_e[sel], marker="o", linestyle="", 
                 linewidth=2, color="maroon", label=r"$t_{exp} \gg$ 3DHST")
    fax.set_xlabel(r"$m_{{{}}}$ (3DHST)".format(ref_band))
    fax.set_ylabel(r"$m_{{{}}}$ (force)".format(ref_band))
    fax.plot(xx, xx, linestyle = "--", color="red", linewidth=2)
    fax.plot(xx, xx-0.2, linestyle=":", color="red", label=r"$\pm 0.2 \, mag$", linewidth=2)
    fax.plot(xx, xx+0.2, linestyle=":", color="red", linewidth=2)
    fax.legend()
    
    fax = faxes[1]
    chi = (mf - mt) / np.sqrt(mf_e**2 + mt_e**2)
    gg = np.isfinite(chi)
    fax.hist(chi[gg], bins=20, alpha=0.5, color="slateblue")
    fax.set_xlabel(r"$\chi = \frac{\Delta m}{\sqrt{\sigma_{force}^2 + \sigma_{3D}^2}}$")   
    fax.text(0.1, 0.7, "std. dev. = {:2.2f}".format(chi.std()), transform=fax.transAxes)
    ffig.savefig("flux_comparison.png", dpi=300)
    
    #sys.exit()
    
    zz = zip(force["patchID"], force["sourceID"], mt, chi)
    
    _ = [print(z) for z in zz]
    
    # --- Plot colors ---
     
    mmax = 26.0
    xx -= 26.0

    b = ("f814w", "f160w")
     
    cf, cf_e = get_color(force, b)
    ch, ch_e = get_color(threed, b)
    g = mt < mmax
     
    cfig, caxes = pl.subplots(1, 1, figsize=(5, 4), squeeze=False)
    cax = caxes[0, 0]
    cax.errorbar(ch, cf, xerr=ch_e, yerr=cf_e, marker="o", linestyle="", 
                 linewidth=2, color="slateblue")
    cax.errorbar(ch[g], cf[g], xerr=ch_e[g], yerr=cf_e[g], marker="o", 
                 linestyle="", linewidth=2, color="forestgreen", 
                 label=r"$F160W_{{3DHST}} < {:3.1f}$".format(mmax))
    cax.set_xlabel("{}-{} (3DHST)".format(*b))
    cax.set_ylabel("{}-{} (force)".format(*b))
    cax.plot(xx, xx, linestyle = "--", color="red", linewidth=2)
    cax.plot(xx, xx-0.2, linestyle=":", color="red", linewidth=2, 
             label=r"$\pm 0.2 \, mag$")
    cax.plot(xx, xx+0.2, linestyle=":", color="red", linewidth=2)
    cax.set_xlim(-2, 3)
    cax.set_ylim(-2, 3)
    cax.legend()
    cfig.savefig("color_comparison.png", dpi=300)