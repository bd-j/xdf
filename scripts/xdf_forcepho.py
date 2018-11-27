import sys

import numpy as np
import matplotlib.pyplot as pl
from scipy import cov
from astropy.io import fits

from forcepho import paths
from forcepho.likelihood import WorkPlan, lnlike_multi

from xdfutils import cat_to_sourcepars, prep_scene, make_xdf_stamp, Posterior, Result
import backends


psfpaths = {"f814w": None,
            "f160w": "gmpsf_hst_f160w_ng3.h5"
            }
imnames = {"f814w": "hlsp_xdf_hst_acswfc-60mas_hudf_f814w_v1_",
           "f160w": "hlsp_xdf_hst_wfc3ir-60mas_hudf_f160w_v1_"
           }
    
# ------------------------------------
# --- Get the MMSE catalog ---
mmse_catname = "/Users/bjohnson/Projects/xdf/data/xdf_f160-f814_3020-3470.fits"
cat = np.array(fits.getdata(mmse_catname))


def setup_patch(xlo, xhi, ylo, yhi, filters=["f160w"]):

    # -- choose a region, get sources in the region ---
    #center = np.array([20., 390.])
    #size = np.array([20., 20.])
    #lo, hi = center - (size/2) - 1, center + (size/2) + 1
    lo = np.array([xlo, ylo])
    hi = np.array([xhi, yhi])
    center = np.round((hi + lo)  / 2)
    size = hi - lo
    sel = ((cat["x"] > lo[0]) & (cat["x"] < hi[0]) &
           (cat["y"] > lo[1]) & (cat["y"] < hi[1]))
    fluxes = [[f*2.0] for f in cat[sel]["flux"]]
    sourcepars = [tuple([flux] + cat_to_sourcepars(s)) for flux, s in zip(fluxes, cat[sel])]

    # --- Setup Scene and Stamp(s) ---

    stamps = [make_xdf_stamp(imnames[f], psfpaths[f], center, size, filtername=f)
              for f in filters]

    return sourcepars, stamps


if __name__ == "__main__":

    # ---------------
    # --- SETUP ---

    corners = 10, 40, 375, 405
    filters = ["f160w"] #, "f814w"]
    nband = len(filters)
    sourcepars, stamps = setup_patch(*corners, filters=filters)
    cx, cy = (corners[0] + corners[1])/2, (corners[2] + corners[3]) / 2
    rname = "results/results_xdf_x{:3.0f}_y{:3.0f}_{}".format(cx, cy, "".join(filters))

    for stamp in stamps:
        #bkg = np.nanmedian(stamp.pixel_values[:5, :])  # stamp.full_header["BKG"]
        bkg = 0.0
        stamp.pixel_values -= bkg # 
        stamp.subtracted_background = bkg

    plans = [WorkPlan(stamp) for stamp in stamps]
    scene, theta = prep_scene(sourcepars, filters=np.unique(filters).tolist(),
                              splinedata=paths.galmixtures[1])

    theta_init = theta.copy()
    p0 = theta_init.copy()
    ndim = len(theta)
    nsource = len(sourcepars)

    # --------------------------------
    # --- Show model and data ---
    if False:
        from phoplot import plot_model_images
        fig, axes = plot_model_images(p0, scene, stamps)
        pl.show()
     
    # --------------------------------
    # --- Priors and scale guesses ---
    plate_scale = np.linalg.eigvals(np.linalg.inv(stamps[0].dpix_dsky))
    plate_scale = np.abs(plate_scale).mean()

    upper = [[15.0, s[1] + 1 * plate_scale, s[2] + 1 * plate_scale, 1.0, np.pi/2, 5.0, 0.3]
             for s in sourcepars]
    lower = [[0.0, s[1] - 1 * plate_scale, s[2] - 1 * plate_scale, 0.3, -np.pi/2, 1.2, 0.015]
             for s in sourcepars]
    upper = np.concatenate(upper)
    lower = np.concatenate(lower)
    fluxes = [s[0] for s in sourcepars]
    scales = np.concatenate([f + [plate_scale, plate_scale, 0.1, 0.1, 0.1, 0.01] for f in fluxes])

    # --------------------------------
    # --- sampling ---

    # --- hemcee ---
    if True:

        result = backends.run_hemcee(p0, scene, plans, scales=scales, nwarm=1000, niter=500)
        
        best = result.chain[result.lnp.argmax(), :]
        result.labels = scene.parameter_names
        result.sourcepars = sourcepars
        result.stamps = stamps
        result.filters = filters

        import cPickle as pickle
        if rname is not None:
            with open("{}_hemcee.pkl".format(rname), "wb") as f:
                pickle.dump(result, f)

        # --- Plotting
        fig, axes = pl.subplots(7+1, len(scene.sources), sharex=True)
        for i, ax in enumerate(axes[:-1, ...].T.flat): ax.plot(result.chain[:, i])
        axes.flatten()[7].plot(result.lnp)
        #for i, ax in enumerate(axes.T.flat): ax.axhline(result.pinitial[i], color='k', linestyle=':')
        for i, ax in enumerate(axes[:-1, ...].T.flat): ax.set_title(result.labels[i])
        import corner
        cfig = corner.corner(result.chain, labels=result.labels,
                             show_titles=True, fill_contours=True,
                             plot_datapoints=False, plot_density=False)

        normchain = (result.chain - result.chain.mean(axis=0)) / result.chain.std(axis=0)
        corr = cov(normchain.T)

    # --- nested ---
    if False:

        result, dr = backends.run_dynesty(scene, plans, nlive=50, lower=lower, upper=upper)

        best = result.chain[result.lnp.argmax(), :]
        result.labels = scene.parameter_names
        result.sourcepars = sourcepars
        result.stamps = stamps
        result.filters = filters

        import cPickle as pickle
        if rname is not None:
            with open("{}_dynesty.pkl".format(rname), "wb") as f:
                pickle.dump(result, f)

        # --- Plotting
        from dynesty import plotting as dyplot
        cfig, caxes = dyplot.cornerplot(dr, fig=pl.subplots(result.ndim, result.ndim, figsize=(13., 10)),
                                        labels=result.labels, show_titles=True, title_fmt='.8f')
        tfig, taxes = dyplot.traceplot(dr, fig=pl.subplots(result.ndim, 2, figsize=(13., 13.)),
                                       labels=result.label)

        
    # ---------------------------
    # --- Plot results ---
    if True:
        # plot the data and model
        from phoplot import plot_model_images
        rfig, raxes = plot_model_images(best, scene, stamps)
        pl.show()
