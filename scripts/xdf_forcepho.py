import sys, os
from functools import partial as argfix
import numpy as np
import matplotlib.pyplot as pl
from scipy import cov

from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.wcs import WCS

from forcepho import paths
from forcepho.sources import Galaxy, Scene
from forcepho.likelihood import WorkPlan, make_image, lnlike_multi
from forcepho.data import PostageStamp
from forcepho import psf as pointspread

from phoutils import Posterior, Result, get_psf


base = "/Users/bjohnson/Projects/xdf/data/images/"

psfpaths = {"f814w": None,
            "f160w": None}
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


def make_stamp(imroot, psfname, center, size, fwhm=3.0, filtername="H"):

    # Values used to produce the MMSE catalog
    im, wght, rms, cutout, wcs = get_cutout(base, root, mmse_position, mmse_size)

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
    stamp.pixel_values = im[xinds, yinds]
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
    stamp.psf = get_psf(psfname, fwhm)

    stamp.filtername = filtername
    return stamp


def prep_scene(sourcepars, filters=["dummy"], splinedata=None, free_sersic=True):

    # --- Get Sources and a Scene -----
    sources = []

    for pars in sourcepars:
        flux, x, y, q, pa, n, rh = np.copy(pars)
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


if __name__ == "__main__":

    # ------------------------------------
    # --- Get the MMSE catalog ---
    catname = "/Users/bjohnson/Projects/xdf/data/xdf_f160-f814_3020-3470.fits"
    cat = np.array(fits.getdata(catname))

    # choose a region
    center = np.array([75., 240.])
    size = np.array([40., 30.])
    lo, hi = center - (size/2) - 1, center + (size/2) + 1
    sel = ((cat["x"] > lo[0]) & (cat["x"] < hi[0]) &
           (cat["y"] > lo[1]) & (cat["y"] < hi[1]))
    fluxes = [[f*2.0] for f in cat[sel]["flux"]]
    sourcepars = [tuple([flux] + cat_to_sourcepars(s)) for flux, s in zip(fluxes, cat[sel])]
    
    # --------------------------------
    # --- Setup Scene and Stamp(s) ---
    hband = "hlsp_xdf_hst_wfc3ir-60mas_hudf_f160w_v1_"
    iband = "hlsp_xdf_hst_acswfc-60mas_hudf_f814w_v1_"
    imnames = [hband] #, iband]
    filters = ["f160w"] #, "f814w"]
    nband = len(filters)
    # psfnames = [psfpaths[f] for f in filters]
    psfnames = ["gmpsf_hst_f160w_ng3.h5"]

    stamps = [make_stamp(root, pname, center, size, filtername=f)
              for root, pname, f in zip(imnames, psfnames, filters)]
    for stamp in stamps:
        bkg = np.nanmedian(stamp.pixel_values[:5, :])  # stamp.full_header["BKG"]
        stamp.pixel_values -= bkg # 
        stamp.subtracted_background = bkg

    plans = [WorkPlan(stamp) for stamp in stamps]
    scene, theta = prep_scene(sourcepars, filters=np.unique(filters).tolist(),
                              splinedata=paths.galmixture)

    theta_init = theta.copy()
    ptrue = theta.copy()
    p0 = ptrue.copy()
    ndim = len(theta)
    nsource = len(sourcepars)

    # --------------------------------
    # --- Show model and data ---
    if True:
        from phoplot import plot_model_images
        fig, axes = plot_model_images(ptrue, scene, stamps)
        pl.show()
     
    # --------------------------------
    # --- Priors ---
    plate_scale = np.linalg.eigvals(np.linalg.inv(stamps[0].dpix_dsky))
    plate_scale = np.abs(plate_scale).mean()

    upper = [[5.0, s[1] + 1 * plate_scale, s[2] + 1 * plate_scale, 1.0, np.pi/2, 5.0, 0.12]
             for s in sourcepars]
    lower = [[0.0, s[1] - 1 * plate_scale, s[2] - 1 * plate_scale, 0.3, -np.pi/2, 1.2, 0.015]
             for s in sourcepars]
    upper = np.concatenate(upper)
    lower = np.concatenate(lower)
    scales = np.concatenate([f + [plate_scale, plate_scale, 0.1, 0.1, 0.1, 0.01] for f in fluxes])

    # --------------------------------
    # --- sampling ---
    import time

    # --- hemcee ---
    if True:
        p0 = ptrue.copy()
        
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
        result.truths = ptrue.copy()
        result.metric = np.copy(metric.variance)
        result.step_size = sampler.step_size.get_step_size()
        with open("sim_sersic_single_hemcee.pkl", "wb") as f:
            pickle.dump(result, f)

        fig, axes = pl.subplots(7, 3, sharex=True)
        for i, ax in enumerate(axes.T.flat): ax.plot(chain[:, i])
        #for i, ax in enumerate(axes.T.flat): ax.axhline(p0[i], color='k', linestyle=':')
        for i, ax in enumerate(axes.T.flat): ax.set_title(label[i])

        normchain = (chain - chain.mean(axis=0)) / chain.std(axis=0)
        corr = cov(normchain.T)

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
        truths = ptrue.copy()
        label = filters + ["ra", "dec", "q", "pa", "n", "rh"]
        cfig, caxes = dyplot.cornerplot(results, fig=pl.subplots(ndim, ndim, figsize=(13., 10)),
                                        labels=label, show_titles=True, title_fmt='.8f', truths=truths)
        tfig, taxes = dyplot.traceplot(results, fig=pl.subplots(ndim, 2, figsize=(13., 13.)),
                                    labels=label)

    # -- hmc ---
    if False:
        p0 = ptrue.copy()

        from hmc import BasicHMC
        model = Posterior(scene, plans, upper=upper, lower=lower, verbose=True)
        hsampler = BasicHMC(model, verbose=False)
        hsampler.ndim = len(p0)

        hsampler.set_mass_matrix(1/scales**2)
        #eps = sampler.find_reasonable_stepsize(p0*1.0)
        #use_eps = eps / 2.0 #1e-2
        use_eps = result.step_size * 2
        print(use_eps)
        result = Result()
        result.step_size = np.copy(use_eps)
        result.metric = scales**2
        #sys.exit()
        
        pos, prob, grad = sampler.sample(pos, iterations=500, mass_matrix=1/scales**2,
                                         epsilon=use_eps, length=20, sigma_length=5,
                                         store_trajectories=True)
        #eps = sampler.find_reasonable_stepsize(pos)
        #pos, prob, grad = sampler.sample(pos, iterations=100, mass_matrix=1/scales**2,
        #                                 epsilon=use_eps, length=30, sigma_length=8,
        #                                 store_trajectories=True)

        best = sampler.chain[sampler.lnp.argmax()]

        result.ndim = len(p0)
        result.sourcepars = sourcepars
        result.stamps = stamps
        result.filters = filters
        result.offsets = None
        result.plans = plans
        result.scene = scene
        result.truths = ptrue.copy()
        result.upper = upper
        result.lower = lower
        result.chain = sampler.chain.copy()
        result.lnp = sampler.lnp.copy()
        result.trajectories = sampler.trajectories
        result.accepted = sampler.accepted
        import cPickle as pickle
        with open("xdf_hmc.pkl", "wb") as f:
            pickle.dump(result, f)

        from phoplot import plot_chain
        out = plot_chain(result, show_trajectories=True, equal_axes=True, source=2)
        #import corner
        #cfig = corner.corner(sampler.chain[10:], truths=ptrue.copy(), labels=label, show_titles=True)
        #sys.exit()
        
    # ---------------------------
    # --- Plot results ---
    if True:
        # plot the data and model
        from phoplot import plot_model_images
        rfig, raxes = plot_model_images(best, scene, stamps)
        pl.show()
