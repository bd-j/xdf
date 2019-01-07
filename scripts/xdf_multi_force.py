import sys, argparse

import numpy as np
import matplotlib.pyplot as pl
from scipy import cov
from astropy.io import fits

from forcepho import paths
from forcepho.likelihood import WorkPlan

from xdfutils import setup_xdf_patch
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
parser.add_argument("--ra", type=float, default=-1,
                    help="central ra of cutout")
parser.add_argument("--dec", type=float, default=0,
                    help="central dec of cutout")
parser.add_argument("--size", type=float, nargs='*', default=[1, 1],
                    help="size in arcsec of cutout")
parser.add_argument("--add_source", type=float, nargs=2, default=[0,0],
                    help=("Add a source by hand offset from the central coordinate "
                          "by the amounts in this argument"))
parser.add_argument("--corners", type=int, nargs=4, default=[10, 40, 375, 405],
                    help="corners [xlo, xhi, ylo, yhi] of MMSE cutout")
parser.add_argument("--filters", type=str, nargs="*", default=["f160w"],
                    help="names of bands to get cutouts for")
parser.add_argument("--nwarm", type=int, default=1000,
                    help="number of iterations for hemcee burn-in")
parser.add_argument("--niter", type=int, default=500,
                    help="number of iterations for hemcee production")
parser.add_argument("--nlive", type=int, default=-1,
                    help="number of dynesty live points")
parser.add_argument("--backend", type=str, default="none",
                    help="Sampling backend to use")
parser.add_argument("--results_name", type=str, default="results/results_xdf",
                    help="root name and path for the output pickle.'none' results in no output.")


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
    nband = len(filters)

    tail = "ra{:.4f}_dec{:.4f}_{}".format(args.ra, args.dec, "".join(filters))

    # ---------------------
    # --- Scene & Stamps ---
    sourcepars, stamps, tail = setup_xdf_patch(args, filters=filters, mmse_cat=cat)
    if args.results_name.lower() != "none":
        rname = "{}_{}_{}".format(args.results_name, tail, args.backend)
    else:
        rname = None

    #if np.any(np.array(args.add_source) != 0):
    #    new = add_source(args.ra + args.add_source[0],
    #                     args.dec + args.add_source[1],
    #                     sourcepars[0][0][0])
    #    sourcepars += new

    plans = [WorkPlan(stamp) for stamp in stamps]
    scene, theta = prep_scene(sourcepars, filters=filters,
                              splinedata=paths.galmixtures[1])

    theta_init = theta.copy()
    p0 = theta_init.copy()
    ndim = len(theta)
    nsource = len(sourcepars)

    # --------------------------------
    # --- Show initial model and data ---
    if False:
        fig, axes = plot_model_images(p0, scene, stamps, share=False)
        pl.show()

    # --------------------------------
    # --- Priors and scale guesses ---
    npix = 3.0
    plate_scale = np.abs(np.linalg.eigvals(np.linalg.inv(stamps[0].dpix_dsky)))
    rh_range = np.array(scene.sources[0].rh_range)
    sersic_range = np.array(scene.sources[0].sersic_range)

    lower = [s.nband * [0.] +
             [s.ra - npix * plate_scale[0], s.dec - npix * plate_scale[1],
              0.3, -np.pi/1.5, sersic_range[0], rh_range[0]]
             for s in scene.sources]
    upper = [(np.array(s.flux) * 10).tolist() +
             [s.ra + npix * plate_scale[0], s.dec + npix * plate_scale[1],
              1.0, np.pi/1.5, sersic_range[-1], rh_range[-1]]
             for s in scene.sources]
    upper = np.concatenate(upper)
    lower = np.concatenate(lower)
    fluxes = np.array([s.flux for s in scene.sources])
    scales = np.concatenate([f.tolist() + [plate_scale[0], plate_scale[1], 0.1, 0.1, 0.1, 0.01]
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
        result.corners = args.corners
        result.ra = args.ra
        result.dec = args.dec
        result.size = args.size

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
