#!/usr/bin/env python

"""
A script to look load a sampled chain and patch data and look at posteriors for
individual objects.

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
            "f435w": ("f_F435W", 25.68),
            "f606w": ("f_F606W", 26.51),
            "f775w": ("f_F775W", 25.69),
            "f814w": ("f_F814Wcand", 25.94),
            "f850lp": ("f_F850LP", 24.87),
            }



path_to_data = "/Users/bjohnson/Projects/xdf/data/"
path_to_results = "/Users/bjohnson/Projects/xdf/results/"
splinedata = pjoin(path_to_data, "sersic_mog_model.smooth=0.0150.h5")
psfpath = pjoin(path_to_data, "psfs", "mixtures")
threedhst_cat = fits.getdata("/Users/bjohnson/Projects/xdf/data/catalogs/goodss_3dhst.v4.1.cat.FITS")
extras = ["ra", "dec", "tot_cor", "a_image", "b_image", "theta_J2000", "kron_radius"]

def query_3dhst(ra, dec, bands, extras=extras):
    hst = SkyCoord(ra=threedhst_cat['ra'], dec=threedhst_cat['dec'], unit=(u.deg, u.deg))
    force = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
    idx, sep, _ = force.match_to_catalog_sky(hst)

    cols = [(b, np.float) for b in bands]
    cols += [("{}_err".format(b), np.float) for b in bands]
    cols += [(n, np.float) for n in extras]
    cols += [("id", np.int)]
    dtype = np.dtype(cols)
    row = np.zeros(1, dtype=dtype)
    catrow = threedhst_cat[idx]
    for n in extras:
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


def plot_images(pos, scene, stamps, source_idx=None, 
                x=slice(None), y=slice(None), dr=1, dd=1,
                axes=None, colorbars=True, share=True,
                scale_model=False, scale_residuals=False, nchi=-1):
    # --- Set up axes ---
    vals = pos
    same_scale = [False, scale_model, scale_residuals, nchi]
    if axes is None:
        figsize = (3.3*len(stamps) + 0.5, 14)
        rfig, raxes = pl.subplots(4, len(stamps), figsize=figsize,
                                  sharex=share, sharey=share)
        raxes = raxes.T
    else:
        rfig, raxes = None, axes
    raxes = np.atleast_2d(raxes)
    
    # --- Restrict to pixels around source ---
    scene.set_all_source_params(vals)
    if source_idx is not None:
        source = scene.sources[source_idx]
        sky = np.array([source.ra, source.dec])
    
    for i, stamp in enumerate(stamps):
        if source_idx is not None:
            xc, yc = stamp.sky_to_pix(sky)
            dx, dy = np.abs(np.dot(stamp.scale, np.array([dr, dd])))
            x = slice(int(xc - dx/2), int(xc + dx/2))
            y = slice(int(yc - dy/2), int(yc + dy/2))

        data = stamp.pixel_values
        model, grad = make_image(scene, stamp, Theta=vals)
        resid = data - model
        chi = resid * stamp.ierr.reshape(stamp.nx, stamp.ny)
        ims = [data, model, resid, chi]
        for j, im in enumerate(ims):
            if (same_scale[j] == 1) and colorbars:
                vmin, vmax = cb.vmin, cb.vmax
            elif (same_scale[j] > 0) and colorbars:
                vmin, vmax = -same_scale[j], same_scale[j]
            else:
                vmin = vmax = None
            ci = raxes[i, j].imshow(im[x, y].T, origin='lower', vmin=vmin, vmax=vmax)
            if colorbars:
                cb = pl.colorbar(ci, ax=raxes[i,j], orientation='horizontal')
                cb.ax.set_xticklabels(cb.ax.get_xticklabels(), fontsize=8, rotation=-50)
        text = stamp.filtername
        ax = raxes[i, 1]
        ax.text(0.6, 0.1, text, transform=ax.transAxes, fontsize=10)

    return rfig, raxes


def show_source(source_id, stamps, scene, chain, pra=0, pdec=0,
                equal_fluxrange=False):
    proposal = chain[-1, :]
    scene.set_all_source_params(proposal)
    
    figsize = (21.5, 15)
    fig, axes = pl.subplots(5, len(stamps), figsize=figsize,
                            sharex='none', sharey="none")
    
    # --- plot images ---
    _, _ = plot_images(proposal, scene, stamps, source_idx=source_id,
                          scale_model=True, scale_residuals=True, 
                          axes=axes[1:, :].T, nchi=5)
    # --- plot flux posterior
    source = scene.sources[source_id]
    ra, dec = source.ra + pra, source.dec + pdec
    sep, threed = query_3dhst(ra, dec, source.filternames)
    start = np.sum([s.nparam for s in scene.sources[:source_id]])
    maxf = []
    
    print(axes.shape)
    for i, stamp in enumerate(stamps):
        ax = axes[0, i]
        ax.set_title(stamp.filtername, fontsize=14)
        ind = int(start + source.filter_index(stamp.filtername))
        samples = chain[:, ind]
        ax.hist(samples, bins=20, alpha=0.5, histtype="stepfilled", color="slateblue")
        ref = threed[stamp.filtername]
        referr = threed["{}_err".format(stamp.filtername)]
        thismax = np.max([ref + referr, samples.max()])
        maxf.append(ref + referr)
        if ref != 0:
            ax.axvline(ref, linestyle="--", color="red")
            yr = ax.get_ylim()
            ax.fill_betweenx(yr, ref-referr, ref+referr, color="red", alpha=0.2)
        ax.set_xlim(0, thismax*1.05)

    if equal_fluxrange:
        [ax.set_xlim(0, np.max(maxf)) for ax in axes[0,:]]

    return fig, axes, threed


def prettify_axes(axes):
    [l.set_rotation(0) for ax in axes[0, :] for l in ax.get_xticklabels()]
    [ax.set_xlabel("counts/s") for ax in axes[0, :]]
    [ax.set_xticklabels([]) for ax in axes[1:-1, :].flat]
    [ax.set_yticklabels([]) for ax in axes[:, 1:].flat]
    labels = ['Flux', 'Data', 'Model', 'Data-Model', "$\chi$"]
    [ax.set_ylabel(labels[i], rotation=60, labelpad=20) for i, ax in enumerate(axes[:, 0])]
    axes[0,0].set_yticklabels([])
    # get_points returns
    #  [[x0, y0],
    #   [x1, y1]]
    
    
    ibb = axes[1,0].get_position().get_points()
    dhi = ibb[1, 1] - ibb[0, 1]
    dwi = ibb[1, 0] - ibb[0, 0]
    
    for ax in axes[0, :]:
        hbb = ax.get_position()
        x0, y0 = hbb.get_points()[0]
        x1, y1 = hbb.get_points()[1]
        ax.set_position([x0, y1-dhi, dwi, dhi])
        
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    force = Patch([], color="slateblue", alpha=0.5)
    td = Line2D([],[], color="red", linestyle="--")
    
    artists = td, force
    labels = ["3DHST", "forcepho"]
    return artists, labels


def get_patch(resultname, splinedata=splinedata, psfpath=psfpath):
    """
    This runs in a single CPU process.  It dispatches the 'patch data' to the
    device and then runs a pymc3 HMC sampler.  Each likelihood call within the
    HMC sampler copies the proposed parameter position to the device, runs the
    kernel, and then copies back the result, returning the summed ln-like and
    the gradients thereof to the sampler.

    :param patchname: 
        Full path to the patchdata hdf5 file.
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


def show_patch(pid, save=True):
    resultname = "results/max10-patch_udf_withcat_{}_result.h5".format(pid)
    stamps, scene, chain, pra, pdec = get_patch(resultname)
    for isource in range(len(scene)):
        fig, axes, threed = show_source(isource, stamps, scene, chain, pra=pra, pdec=pdec)
        artists, legends = prettify_axes(axes)
        idx = threed[0]["id"]
        ra, dec = threed[0]["ra"], threed[0]["dec"]
        fig.suptitle("Patch: {}, 3DHST ID: {}, RA:{:3.6f}, Dec: {:3.5f}".format(pid, idx, ra, dec))
        
        fig.legend(artists, legends, loc='upper right', ncol=len(artists), bbox_to_anchor=(0.5, 0.85), frameon=True)
        
        if save:
            figname = "sourcefig/patch{:03.0f}_source{:02.0f}_3dhst{}.pdf".format(pid, isource, idx)
            fig.savefig(figname)
        pl.close(fig)
    return resultname

if __name__ == "__main__":
    pids = [90, 91, 157, 159, 183, 274, 382, 653]
    #pid = 159
    for pid in pids:
        rname = show_patch(pid)