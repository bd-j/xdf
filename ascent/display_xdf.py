#!/usr/bin/env python

"""
A script to read in many XDF patches and launch a sampler on each patch

On Ascent with CUDA MPS, one could run this with 16 processes on 7 cores using 1 GPU with:
$ jsrun -n1 -a16 -c7 -g1 ./run_patch_gpu_test.py
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

_print = print
print = lambda *args,**kwargs: _print(*args,**kwargs, file=sys.stderr, flush=True)

def plot_chain(result):
    npar = np.max([s.nparam for s in result.scene.sources])
    
    figsize = (4 * len(result.scene.sources) + 1, 12)
    fig, axes = pl.subplots(npar+1, len(result.scene.sources), sharex=True, figsize=figsize)
    for i, ax in enumerate(axes[:-1, ...].T.flat):
        ax.plot(result.chain[:, i])
        ax.set_ylabel(result.scene.parameter_names[i])
    try:
        [ax.plot(result.lnp) for ax in axes[-1, ...].flat]
        [ax.set_xlabel("iteration") for ax in axes[-1, ...].flat]
        axes[-1, ...].set_ylabel("ln P")
    except(AttributeError):
        [ax.set_xlabel("iteration") for ax in axes[-2, ...].flat]
        [ax.set_visible(False) for ax  in axes[-1, ...].flat]

    return fig, axes


def plot_model_images(pos, scene, stamps, axes=None, colorbars=True,
                      x=slice(None), y=slice(None), share=True,
                      scale_model=False, scale_residuals=False, nchi=-1):
    vals = pos
    same_scale = [False, scale_model, scale_residuals, nchi]
    if axes is None:
        rfig, raxes = pl.subplots(4, len(stamps), figsize=(3.3*len(stamps) + 0.5, 14),
                                  sharex=share, sharey=share)
        raxes = raxes.T
    else:
        rfig, raxes = None, axes
    raxes = np.atleast_2d(raxes)
    for i, stamp in enumerate(stamps):
        data = stamp.pixel_values
        im, grad = make_image(scene, stamp, Theta=vals)
        resid = data - im
        chi = resid * stamp.ierr.reshape(stamp.nx, stamp.ny)
        ims = [data, im, resid, chi]
        for j, ii in enumerate(ims):
            ax = raxes[i, j]
            if (same_scale[j] == 1) and colorbars:
                vmin, vmax = cb.vmin, cb.vmax
            elif (same_scale[j] > 0) and colorbars:
                vmin, vmax = -same_scale[j], same_scale[j]
            else:
                vmin = vmax = None
            ci = ax.imshow(ii[x, y].T, origin='lower', vmin=vmin, vmax=vmax)
            _ = [tick.label.set_fontsize(10) 
                 for tick in ax.xaxis.get_major_ticks()]
            _ = [tick.label.set_fontsize(10) 
                 for tick in ax.yaxis.get_major_ticks()]
            if (rfig is not None) & colorbars:
                cb = rfig.colorbar(ci, ax=raxes[i,j], orientation='horizontal')
                cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=-55, fontsize=8)
        #text = "{}\n({}, {})".format(stamp.filtername, stamp.crval[0], stamp.crval[1])
        text = stamp.filtername
        ax = raxes[i, 1]
        ax.text(0.6, 0.1, text, transform=ax.transAxes, fontsize=10)
        raxes[i, 0].set_title(text, fontsize=14)

    labels = ['Data', 'Model', 'Data-Model', r"$\chi$"]
    _ = [ax.set_ylabel(labels[i], rotation=60, labelpad=20) 
         for i, ax in enumerate(raxes[0,:])]
    return rfig, raxes



path_to_data = "/Users/bjohnson/Projects/xdf/data/"
path_to_results = "/Users/bjohnson/Projects/xdf/results/"
#patch_name = pjoin(path_to_data, "test_patch_mini.h5")  # "test_patch_large.h5" or test_patch.h5 or "test_patch_mini.h5"
splinedata = pjoin(path_to_data, "sersic_mog_model.smooth=0.0150.h5")
psfpath = pjoin(path_to_data, "psfs", "mixtures")

def show_patch(resultname, splinedata=splinedata, psfpath=psfpath, maxactive=2):
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
    if os.path.exists(resultname):
        with h5py.File(resultname, "r") as f:
            chain = f["chain"][:]
            patch = f.attrs["patchname"]
            pra, pdec = tuple(f.attrs["sky_reference"])
            ncall = f.attrs["ncall"]
            twall = f.attrs["wall_time"]
            lower = f.attrs["lower_bounds"]
            upper = f.attrs["upper_bounds"]
            nsources = f.attrs["nactive"]
        proposal = chain[48, :]
    else:
        patch = os.path.basename(resultname).replace("_result", "")
        chain, lower, upper = None, None, None
        proposal = None
        nsources = 0
        pra = pdec = None
        
    print("average call time = {}s".format(twall / ncall))

    patchname = pjoin(path_to_data, "patches", "20190612_9x9_threedhst", 
                      os.path.basename(patch))

    # --- Prepare Patch Data ---
    stamps, miniscene = patch_conversion(patchname, splinedata, psfpath, nradii=9)
    miniscene = set_inactive(miniscene, [stamps[0], stamps[-1]], nmax=nsources)
    if pra is not None:
        zerocoords(stamps, miniscene, sky_zero=np.array([pra, pdec]))
    p0 = miniscene.get_all_source_params().copy()
    if proposal is None:
        proposal = p0

    # plot the thing
    rfig, raxes = plot_model_images(proposal, miniscene, stamps, 
                                    scale_model=True, scale_residuals=True, nchi=5, 
                                    share="col")
    return rfig, raxes, (p0, chain, miniscene, lower, upper)


def show_one(pid):

    resultname = "results/max10-patch_udf_withcat_{}_result.h5".format(pid)
    rfig, rax, blob = show_patch(resultname, maxactive=2)
    p0, chain, scene, lower, upper = blob
    print(len(scene))

    result = Result()
    result.scene = scene
    result.chain = chain
    if chain is not None:
        cfig, cax = plot_chain(result)
        
    return rfig, cfig
    

if __name__ == "__main__":
    
    plot_dir = "sourcefig"
    try:
        pids = [sys.argv[1]]
    except(IndexError):
        pids = [90, 91, 653, 382, 274, 183, 159, 157]
        
    for pid in pids:
        print("Patch {}".format(pid))
        rfig, cfig = show_one(pid)
        if len(pids) == 1:
            continue
        
        rfig.savefig("{}/patch{:03.0f}_all.pdf".format(plot_dir, pid))
        cfig.savefig("{}/patch{:03.0f}_all.chain.pdf".format(plot_dir, pid))
        pl.close("all")
