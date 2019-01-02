import sys, os, time
from functools import partial as argfix
import numpy as np
import matplotlib.pyplot as pl

from forcepho.likelihood import WorkPlan, lnlike_multi
from xdfutils import Posterior, Result, LogLikeWithGrad


__all__ = ["run_hemcee", "run_dynesty", "run_hmc"]


def priors(stamps, sourcepars):
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
    fluxes = [s[0] for s in sourcepars]
    scales = np.concatenate([f + [plate_scale, plate_scale, 0.1, 0.1, 0.1, 0.01] for f in fluxes])

    return scales, lower, upper


def run_pymc3(p0, scene, plans, lower=-np.inf, upper=np.inf,
              priors=None, nwarm=2000, niter=1000):

    import pymc3 as pm
    import theano.tensor as tt

    model = Posterior(scene, plans)
    logl = LogLikeWithGrad(model)
    pnames = scene.parameter_names
    t = time.time()
    with pm.Model() as opmodel:
        if priors is None:
            z0 = [pm.Uniform(p, lower=l, upper=u)
                for p, l, u in zip(pnames, lower, upper)]
        else:
            z0 = priors
        theta = tt.as_tensor_variable(z0)
        pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
        trace = pm.sample(draws=niter, tune=nwarm, cores=1, discard_tuned_samples=True)

    tsample = time.time() - t
    
    result = Result()
    result.ndim = len(p0)
    result.pinitial = p0.copy()
    result.chain = np.array([trace.get_values(n) for n in pnames]).T
    result.trace = trace
    #result.lnp = sampler.lnp.copy()
    result.ncall = model.ncall
    result.wall_time = tsample
    result.plans = plans
    result.scene = scene
    result.lower = lower
    result.upper = upper

    return result


def run_hemcee(p0, scene, plans, scales=1.0, nwarm=2000, niter=1000):
    
    # --- hemcee ---

    from hemcee import NoUTurnSampler
    from hemcee.metric import DiagonalMetric
    metric = DiagonalMetric(scales**2)
    model = Posterior(scene, plans, upper=np.inf, lower=-np.inf)
    sampler = NoUTurnSampler(model.lnprob, model.lnprob_grad, metric=metric)

    result = Result()
    result.ndim = len(p0)
    result.pinitial = p0.copy()
    
    t = time.time()
    pos, lnp0 = sampler.run_warmup(p0, nwarm)
    twarm = time.time() - t
    ncwarm = np.copy(model.ncall)
    model.ncall = 0
    t = time.time()
    chain, lnp = sampler.run_mcmc(pos, niter)
    tsample = time.time() - t
    ncsample = np.copy(model.ncall)

    result.chain = chain
    result.lnp = lnp
    result.ncall = (ncwarm, ncsample)
    result.wall_time = (twarm, tsample)
    result.plans = plans
    result.scene = scene
    result.metric_variance = np.copy(metric.variance)
    result.step_size = sampler.step_size.get_step_size()

    return result


def run_dynesty(scene, plans, lower=0, upper=1.0, nlive=50):

    # --- nested ---
    lnlike = argfix(lnlike_multi, scene=scene, plans=plans, grad=False)
    theta_width = (upper - lower)
    ndim = len(lower)
    
    def prior_transform(unit_coords):
        # now scale and shift
        theta = lower + theta_width * unit_coords
        return theta

    import dynesty
    sampler = dynesty.DynamicNestedSampler(lnlike, prior_transform, ndim, nlive=nlive,
                                           bound="multi", method="slice", bootstrap=0)
    t0 = time.time()
    sampler.run_nested(nlive_init=int(nlive/2), nlive_batch=int(nlive),
                       wt_kwargs={'pfrac': 1.0}, stop_kwargs={"post_thresh":0.2})
    tsample = time.time() - t0

    dresults = sampler.results
    
    result = Result()
    result.ndim = ndim
    result.chain = dresults["samples"]
    result.lnp = dresults['logl']
    #result.ncall = nsample
    result.wall_time = tsample
    result.plans = plans
    result.scene = scene
    result.lower = lower
    result.upper = upper

    return result, dresults


def run_hmc(p0, scene, plans, scales=1.0, lower=-np.inf, upper=np.inf,
            nwarm=0, niter=500, length=20):

    # -- hmc ---
    from hmc import BasicHMC
    model = Posterior(scene, plans, upper=upper, lower=lower, verbose=True)
    hsampler = BasicHMC(model, verbose=False)
    hsampler.ndim = len(p0)
    hsampler.set_mass_matrix(1/scales**2)

    result = Result()
    result.pinitial = p0.copy()
    
    eps = sampler.find_reasonable_stepsize(p0*1.0)
    use_eps = result.step_size * 2
    result.step_size = np.copy(use_eps)
    result.metric = scales**2

    if nwarm > 0:
        pos, prob, grad = sampler.sample(pos, iterations=nwarm, mass_matrix=1/scales**2,
                                        epsilon=use_eps, length=length, sigma_length=int(length/4),
                                        store_trajectories=True)
        use_eps = sampler.find_reasonable_stepsize(pos)
        result.step_size = np.copy(use_eps)
        ncwarm = np.copy(model.ncall)
        model.ncall = 0

    pos, prob, grad = sampler.sample(pos, iterations=niter, mass_matrix=1/scales**2,
                                     epsilon=use_eps, length=length, sigma_length=int(length/4),
                                     store_trajectories=True)

    result.ndim = len(p0)
    result.chain = sampler.chain.copy()
    result.lnp = sampler.lnp.copy()
    result.plans = plans
    result.scene = scene
    result.lower = lower
    result.upper = upper
    result.trajectories = sampler.trajectories
    result.accepted = sampler.accepted

    return result
