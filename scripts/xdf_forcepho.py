import sys
from functools import partial as argfix
import numpy as np
import matplotlib.pyplot as pl
from scipy import cov

from astropy.io import fits

from forcepho import paths
from forcepho.likelihood import WorkPlan, lnlike_multi
from xdfutils import cat_to_sourcepars, prep_scene, make_xdf_stamp, Posterior, Result



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
    import time

    # --- hemcee ---
    if True:
        
        from hemcee import NoUTurnSampler
        from hemcee.metric import DiagonalMetric
        metric = DiagonalMetric(scales**2)
        model = Posterior(scene, plans, upper=np.inf, lower=-np.inf)
        sampler = NoUTurnSampler(model.lnprob, model.lnprob_grad, metric=metric)

        t = time.time()
        pos, lnp0 = sampler.run_warmup(p0, 2000)
        twarm = time.time() - t
        nwarm = np.copy(model.ncall)
        model.ncall = 0
        t = time.time()
        chain, lnp = sampler.run_mcmc(pos, 1000)
        tsample = time.time() - t
        nsample = np.copy(model.ncall)
        best = chain[lnp.argmax(), :]
        label = np.concatenate([s.parameter_names for s in scene.sources])

        import cPickle as pickle
        result = Result()
        result.ndim = len(p0)
        result.chain = chain
        result.lnp = lnp
        result.ncall = nsample
        result.wall_time = tsample
        result.sourcepars = sourcepars
        result.stamps = stamps
        result.filters = filters
        result.plans = plans
        result.scene = scene
        result.truths = theta_init.copy()
        result.metric = np.copy(metric.variance)
        result.step_size = sampler.step_size.get_step_size()
        if rname is not None:
            with open("{}_hemcee.pkl".format(rname), "wb") as f:
                pickle.dump(result, f)

        fig, axes = pl.subplots(7+1, len(scene.sources), sharex=True)
        for i, ax in enumerate(axes[:-1, ...].T.flat): ax.plot(chain[:, i])
        axes.flatten()[7].plot(lnp)
        #for i, ax in enumerate(axes.T.flat): ax.axhline(p0[i], color='k', linestyle=':')
        for i, ax in enumerate(axes[:-1, ...].T.flat): ax.set_title(label[i])

        normchain = (chain - chain.mean(axis=0)) / chain.std(axis=0)
        corr = cov(normchain.T)
        import corner
        label = scene.parameter_names
        cfig = corner.corner(chain, labels=label,
                             show_titles=True, fill_contours=True,
                             plot_datapoints=False, plot_density=False)
        

    # --- nested ---
    if False:
        lnlike = argfix(lnlike_multi, scene=scene, plans=plans, grad=False)
        theta_width = (upper - lower)
        nlive = 50
        
        def prior_transform(unit_coords):
            # now scale and shift
            theta = lower + theta_width * unit_coords
            return theta

        import dynesty
        
        # "Standard" nested sampling.
        sampler = dynesty.DynamicNestedSampler(lnlike, prior_transform, ndim, nlive=nlive,
                                               bound="multi", method="slice", bootstrap=0)
        t0 = time.time()
        sampler.run_nested(nlive_init=int(nlive/2), nlive_batch=int(nlive),
                           wt_kwargs={'pfrac': 1.0}, stop_kwargs={"post_thresh":0.2})
        dur = time.time() - t0
        results = sampler.results
        results['duration'] = dur
        indmax = results['logl'].argmax()
        best = results['samples'][indmax, :]

        from dynesty import plotting as dyplot
        truths = theta_init.copy()
        label = filters + ["ra", "dec", "q", "pa", "n", "rh"]
        cfig, caxes = dyplot.cornerplot(results, fig=pl.subplots(ndim, ndim, figsize=(13., 10)),
                                        labels=label, show_titles=True, title_fmt='.8f', truths=truths)
        tfig, taxes = dyplot.traceplot(results, fig=pl.subplots(ndim, 2, figsize=(13., 13.)),
                                    labels=label)

        
    # ---------------------------
    # --- Plot results ---
    if True:
        # plot the data and model
        from phoplot import plot_model_images
        rfig, raxes = plot_model_images(best, scene, stamps)
        pl.show()
