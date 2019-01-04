import sys, argparse

import numpy as np
import matplotlib.pyplot as pl
from scipy import cov
from astropy.io import fits

from forcepho import paths
from forcepho.likelihood import WorkPlan
from forcepho.posterior import Posterior

from xdfutils import cat_to_sourcepars, prep_scene, xdf_pixel_stamp, Result
import backends
from phoplot import plot_model_images, display


psfpaths = {"f814w": "../data/psfs/mixtures/gmpsf_30mas_hst_f814w_ng4.h5",
            "f160w": "../data/psfs/mixtures/gmpsf_hst_f160w_ng3.h5"
            }
imnames = {"f814w": "hlsp_xdf_hst_acswfc-30mas_hudf_f814w_v1_",
           "f160w": "hlsp_xdf_hst_wfc3ir-60mas_hudf_f160w_v1_"
           }
    
# ------------------------------------
# --- Get the MMSE catalog ---
mmse_catname = "/Users/bjohnson/Projects/xdf/data/catalogs/xdf_f160-f814_3020-3470.fits"
cat = np.array(fits.getdata(mmse_catname))


# --------------------------------
# --- Command Line Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--xlo", type=int, default=-1,
                    help="low x pixel coordinate of MMSE cutout")
parser.add_argument("--xhi", type=int, default=-1,
                    help="high x pixel coordinate of MMSE cutout")
parser.add_argument("--ylo", type=int, default=-1,
                    help="low y pixel coordinate of MMSE cutout")
parser.add_argument("--yhi", type=int, default=-1,
                    help="high y pixel coordinate of MMSE cutout")
parser.add_argument("--corners", type=int, nargs=4, default=[10, 40, 375, 405],
                    help="corners [xlo, xhi, ylo, yhi] of MMSE cutout")
parser.add_argument("--add_source", type=float, nargs=2, default=[0,0],
                    help=("Add a source by hand offset from the lower left "
                          "corner by the amounts in this argument"))
parser.add_argument("--filters", type=list, default=["f160w"],
                    help="names of bands to get cutouts for")
parser.add_argument("--nwarm", type=int, default=1000,
                    help="number of iterations for hemcee burn-in")
parser.add_argument("--niter", type=int, default=500,
                    help="number of iterations for hemcee production")
parser.add_argument("--nlive", type=int, default=-1,
                    help="number of dynesty live points")
parser.add_argument("--backend", type=str, default="hemcee",
                    help="Sampling backend to use")
parser.add_argument("--results_name", type=str, default="results/results_xdf",
                    help="root name and path for the output pickle.")


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

    stamps = [xdf_pixel_stamp(imnames[f], psfpaths[f], center, size, filtername=f)
              for f in filters]

    return sourcepars, stamps


def add_source(ra, dec, primary_flux):
    """Add a dummy source by hand
    """
    spars = ([primary_flux / 10.], ra, dec, 0.9, 0.0, 2, 0.1)
    return [spars]


if __name__ == "__main__":

    # ---------------
    # --- SETUP ---

    # ---------------------
    # --- variables and file names ---
    args = parser.parse_args()
    filters = args.filters
    filters = ["f160w"] #, "f814w"]
    nband = len(filters)

    if args.xlo <= 0:
        corners = tuple(args.corners)
    else:
        corners = args.xlo, args.xhi, args.ylo, args.yhi
    print(corners)
    cx, cy = (corners[0] + corners[1])/2, (corners[2] + corners[3]) / 2
    tail = "x{:.0f}_y{:.0f}_{}".format(cx, cy, "".join(filters))
    if args.results_name.lower() != "none":
        rname = "{}_{}_{}".format(args.results_name, tail, args.backend)
    else:
        rname = None

    # ---------------------
    # --- Scene & Stamps ---
    sourcepars, stamps = setup_patch(*corners, filters=filters)
    if np.any(args.add_source != 0):
        new = add_source(corners[0] + args.add_source[0],
                         corners[2] + args.add_source[1],
                         sourcepars[0][0][0])
        sourcepars += new

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
    # --- Show initial model and data ---
    if False:
        fig, axes = plot_model_images(p0, scene, stamps)
        pl.show()
     
    # --------------------------------
    # --- Priors and scale guesses ---
    plate_scale = np.linalg.eigvals(np.linalg.inv(stamps[0].dpix_dsky))
    plate_scale = np.abs(plate_scale).mean()
    npix = 0.5
    rh_range = np.array(scene.sources[0].rh_range)
    sersic_range = np.array(scene.sources[0].sersic_range)
    npar = scene.sources[0].nparam
    flux_ranges = np.array([[0, s[0][0] * 10] for s in sourcepars])

    lower = [[fr[0], s[1] - npix * plate_scale, s[2] - npix * plate_scale,
              0.3, -np.pi/2, sersic_range[0], rh_range[0]]
             for s, fr in zip(sourcepars, flux_ranges)]
    upper = [[fr[1], s[1] + npix * plate_scale, s[2] + npix * plate_scale,
              1.0, np.pi/2, sersic_range[-1], rh_range[-1]]
             for s, fr in zip(sourcepars, flux_ranges)]
    upper = np.concatenate(upper)
    lower = np.concatenate(lower)
    fluxes = [s[0] for s in sourcepars]
    scales = np.concatenate([f + [plate_scale, plate_scale, 0.1, 0.1, 0.1, 0.01]
                             for f in fluxes])

    # --------------------------------
    # --- sampling ---
    
    # --- pymc3 ---
    if args.backend == "pymc3":

        result = backends.run_pymc3(p0, scene, plans, lower=lower, upper=upper,
                                    nwarm=args.nwarm, niter=args.niter)

        result.labels = scene.parameter_names
        result.sourcepars = sourcepars
        result.stamps = stamps
        result.filters = filters
        result.corners = corners
        best = result.chain[-1, :]

        import cPickle as pickle
        if rname is not None:
            trace = result.trace
            result.trace = None
            with open("{}.pkl".format(rname), "wb") as f:
                pickle.dump(result, f)
            result.trace = trace
                
        # --- Plotting
        _ = display(result, save=False, show=True)
        normchain = (result.chain - result.chain.mean(axis=0)) / result.chain.std(axis=0)
        corr = cov(normchain.T)

    # --- hemcee ---
    if args.backend == "hemcee":

        result = backends.run_hemcee(p0, scene, plans, scales=scales,
                                     nwarm=args.nwarm, niter=args.niter)
        
        best = result.chain[result.lnp.argmax(), :]
        result.labels = scene.parameter_names
        result.sourcepars = sourcepars
        result.stamps = stamps
        result.filters = filters
        result.corners = corners

        import cPickle as pickle
        if rname is not None:
            with open("{}.pkl".format(rname), "wb") as f:
                pickle.dump(result, f)

        # --- Plotting
        _ = display(result, save=False, show=True)
        normchain = (result.chain - result.chain.mean(axis=0)) / result.chain.std(axis=0)
        corr = cov(normchain.T)

    # --- nested ---
    if args.backend == "dynesty":

        result, dr = backends.run_dynesty(scene, plans, lower=lower, upper=upper,
                                          nlive=args.nlive)

        best = result.chain[result.lnp.argmax(), :]
        result.labels = scene.parameter_names
        result.sourcepars = sourcepars
        result.stamps = stamps
        result.filters = filters

        import cPickle as pickle
        if rname is not None:
            with open("{}.pkl".format(rname), "wb") as f:
                pickle.dump(result, f)

        # --- Plotting
        from dynesty import plotting as dyplot
        cfig, caxes = dyplot.cornerplot(dr, fig=pl.subplots(result.ndim, result.ndim, figsize=(13., 10)),
                                        labels=result.labels, show_titles=True, title_fmt='.8f')
        tfig, taxes = dyplot.traceplot(dr, fig=pl.subplots(result.ndim, 2, figsize=(13., 13.)),
                                       labels=result.label)
        rfig, raxes = plot_model_images(best, result.scene, result.stamps)

    # --- No fitting ---
    if args.backend == "none":
        sys.exit()
