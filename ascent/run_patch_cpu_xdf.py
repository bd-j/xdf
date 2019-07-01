#!/usr/bin/env python

"""
A script to read in many XDF patches and launch a sampler on each patch

On Ascent with CUDA MPS, one could run this with 16 processes on 7 cores using 1 GPU with:
$ jsrun -n1 -a16 -c7 -g1 ./run_patch_gpu_test.py
"""

import sys, os
from time import time
from os.path import join as pjoin
import numpy as np
from scipy.optimize import minimize
try:
    import cPickle as pickle
except(ImportError):
    import pickle

from forcepho.posterior import LogLikeWithGrad
from forcepho.likelihood import lnlike_multi, make_image, WorkPlan
from forcepho.fitting import Result

from patch_conversion import patch_conversion, zerocoords

import theano
import pymc3 as pm
import theano.tensor as tt
theano.gof.compilelock.set_lock_status(False)
# be quiet
import logging
logger = logging.getLogger("pymc3")
logger.propagate = False
logger.setLevel(logging.ERROR)


_print = print
print = lambda *args,**kwargs: _print(*args,**kwargs, file=sys.stderr, flush=True)


def save_results(result, rname):
   if rname is not None:
        with open(rname, "wb") as f:
            pickle.dump(result, f)


class CPUPosterior:

    def __init__(self, stamps=[], scene=None, lnlike=lnlike_multi, 
                 lnlike_kwargs={"source_meta": True}):
        self.lnlike = lnlike
        self.lnlike_kwargs = lnlike_kwargs
        self.plans = [WorkPlan(stamp) for stamp in stamps]
        self.scene = scene
        self.ncall = 0
        self._z = -99

    def evaluate(self, z):
        """
        :param z: 
            The untransformed (sampling) parameters which have a prior
            distribution attached.

        Theta are the transformed forcepho native parameters.  In the default
        case these are these are the same as thetaprime, i.e. the transformation
        is the identity.
        """
        Theta = z
        ll, ll_grad = self.lnlike(Theta, scene=self.scene, plans=self.plans,
                                  **self.lnlike_kwargs)

        self.ncall += 1
        self._lnp = ll
        self._lnp_grad = ll_grad
        self._z = z

    def lnprob(self, z):
        if np.any(z != self._z):
            self.evaluate(z)
        return self._lnp

    def lnprob_grad(self, z):
        if np.any(z != self._z):
            self.evaluate(z)
        return self._lnp_grad
    
    def nll(self, z):
        if np.any(z != self._z):
            self.evaluate(z)
        return -self._lnp, -self._lnp_grad
    
    def residuals(self, z):
        
        self.scene.set_all_source_params(z)
        self.residuals = [make_image(self.scene, plan) for plan in self.plans]
        return self.residuals



def prior_bounds(scene, npix=5, flux_factor=20):
    """ Priors and scale guesses:
    
    :param scene:
        A Scene object.  Each source must have the `stamp_cds` attribute.
        
    :param npix: (optional, default: 3)
        Number of pixels to adopt as the positional prior
    """
    dpix_dsky = scene.sources[0].stamp_cds[0]
    source = scene.sources[0]
    plate_scale = np.abs(np.linalg.eigvals(np.linalg.inv(dpix_dsky)))
    rh_range = np.array(scene.sources[0].rh_range)
    sersic_range = np.array(scene.sources[0].sersic_range)

    lower = [s.nband * [0.] +
             [s.ra - npix * plate_scale[0], s.dec - npix * plate_scale[1],
              0.3, -np.pi/1.5, sersic_range[0], rh_range[0]]
             for s in scene.sources]
    upper = [(np.array(s.flux) * flux_factor).tolist() +
             [s.ra + npix * plate_scale[0], s.dec + npix * plate_scale[1],
              1.0, np.pi/1.5, sersic_range[-1], rh_range[-1]]
             for s in scene.sources]
    lower = np.concatenate(lower)
    upper = np.concatenate(upper)

    return lower, upper


path_to_data = "/Users/bjohnson/Projects/xdf/data/"
path_to_results = "/Users/bjohnson/Projects/xdf/results/"
#patch_name = pjoin(path_to_data, "test_patch_mini.h5")  # "test_patch_large.h5" or test_patch.h5 or "test_patch_mini.h5"
splinedata = pjoin(path_to_data, "sersic_mog_model.smooth=0.0150.h5")
psfpath = pjoin(path_to_data, "psfs", "mixtures")

def run_patch(patchname, splinedata=splinedata, psfpath=psfpath, 
              nwarm=200, niter=500, runtype="timing", ntime=1):
    """
    This runs in a single CPU process.  It dispatches the 'patch data' to the
    device and then runs a pymc3 HMC sampler.  Each likelihood call within the
    HMC sampler copies the proposed parameter position to the device, runs the
    kernel, and then copies back the result, returning the summed ln-like and
    the gradients thereof to the sampler.

    :param patchname: 
        Full path to the patchdata hdf5 file.
    """

    print(patchname)

    resultname = os.path.basename(patchname).replace("h5", "pkl")
    resultname = pjoin(path_to_results, resultname)
    try:
        r = pool.rank
    except:
        r = 0
    print("Rank {} writing to {}".format(r, resultname))

    # --- Prepare Patch Data ---
    stamps, miniscene = patch_conversion(patchname, splinedata, psfpath, nradii=9)
    pra = np.median([s.ra for s in miniscene.sources])
    pdec = np.median([s.dec for s in miniscene.sources])
    zerocoords(stamps, miniscene, sky_zero=np.array([pra, pdec]))
    p0 = miniscene.get_all_source_params().copy()


    # --- Instantiate the ln-likelihood object ---
    # This object splits the lnlike_function into two, since that computes 
    # both lnp and lnp_grad, and we need to wrap them in separate theano ops.
    model = CPUPosterior(stamps, miniscene)

    # --- Subtract off the fixed sources ---
    # TODO

    if runtype == "sample":
        # -- Launch HMC ---
        # wrap the loglike and grad in theano tensor ops
        logl = LogLikeWithGrad(model)
        # Get upper and lower bounds for variables
        lower, upper = prior_bounds(miniscene)
        print(lower.dtype, upper.dtype)
        pnames = miniscene.parameter_names
        # The pm.sample() method below will draw an initial theta, 
        # then call logl.perform and logl.grad multiple times
        # in a loop with different theta values.
        t = time()
        with pm.Model() as opmodel:
            # set priors for each element of theta
            z0 = [pm.Uniform(p, lower=l, upper=u) 
                  for p, l, u in zip(pnames, lower, upper)]
            theta = tt.as_tensor_variable(z0)
            # instantiate target density and start sampling.
            pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
            trace = pm.sample(draws=niter, tune=nwarm, progressbar=False,
                              cores=1, discard_tuned_samples=True)

        ts = time() - t
        # yuck.
        chain = np.array([trace.get_values(n) for n in pnames]).T
        
        result = Result()
        result.ndim = len(p0)
        result.pinitial = p0.copy()
        result.chain = chain
        #result.trace = trace
        result.ncall = np.copy(model.ncall)
        result.wall_time = ts
        result.scene = miniscene
        result.lower = np.array(lower)
        result.upper = np.array(upper)
        result.patchname = patchname
        result.sky_reference = (pra, pdec)
        
        #last = chain[:, -1]
        #model.proposer.patch.return_residuals = True
        #result.residuals = model.residuals(last)

        save_results(result, resultname)

    elif runtype == "optimize":
        # --- Launch an optimization ---
        opts = {'ftol': 1e-6, 'gtol': 1e-6, 'factr': 10.,
                'disp':False, 'iprint': 1, 'maxcor': 20}
        theta0 = miniscene.get_all_source_params().copy()
        t = time()
        scires = minimize(model.nll, theta0, jac=True,  method='BFGS',
                          options=opts, bounds=None)
        ts = time() - t
        chain = scires       

    elif runtype == "timing":
        # --- Time a single call ---
        t = time()
        for i in range(ntime):
            model.evaluate(p0)
        ts = time() - t
        chain = [model._lnp, model._lnp_grad]
        print("took {}s for a single call".format(ts / ntime))
    
    return chain, (r, model.ncall, ts)

# Create an MPI process pool.  Each worker process will sit here waiting for input from master
# after it is done it will get the next value from master.
#try:
#    from emcee.utils import MPIPool
#    pool = MPIPool(debug=False, loadbalance=False)
#    if not pool.is_master():
        # Wait for instructions from the master process.
#        pool.wait()
#        sys.exit(0)
#except(ImportError, ValueError):
#    pool = None
#    print('Not using MPI')


def halt(message):
    """Exit, closing pool safely.
    """
    print(message)
    try:
        pool.close()
    except:
        pass
    sys.exit(0)


if __name__ == "__main__":
    patches = [1, 100, 101]
    allpatches = [pjoin(path_to_data, "patches", "20190612_9x9_threedhst", "patch_udf_withcat_{}.h5".format(pid)) 
                  for pid in patches]
    print(allpatches)
    #try:
    #    M = pool.map
    #except:
    #    M = map

    M = map
    t = time()
    #allchains = M(run_patch, allpatches)
    chain = run_patch(allpatches[0], ntime=10)
    twall = time() - t

    halt("finished all patches in {}s".format(twall))
