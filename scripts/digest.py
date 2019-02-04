import os, sys, glob
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as pl

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits

from phoplot import display

# get: region, nsource, npix, twall, nlike.
# get: fluxes, unc_flux, color, unc_color
# get: n_sources < rh
# get: rh, n, pa, q and uncertainties (don't forget to mod(q, pi/2))


# --- Possible to dos ----
# identify failure modes in detail
# fit bands separately (for comparison)
# add more bands
# do more sources
# complexify model (priors, additional sources)

def rectify_pa(phi):
    phim = np.mod((phi + np.pi / 2), np.pi) - np.pi / 2.
    return phim


def quantile(data, percents, weights=None):
    ''' percents in units of 1%
    weights specifies the frequency (count) of data.
    '''
    if weights is None:
        return np.percentile(data, percents)
    ind = np.argsort(data)
    d, w = data[ind], weights[ind]
    p = 1.*w.cumsum()/w.sum()*100
    y = np.interp(percents, p, d)
    return y


def get_scene_stats(chain, parnames, percents=[50, 16, 84]):
    stats = []
    for c, p in zip(chain.T, parnames):
        if "pa" in p:
            c = rectify_pa(c)
        par, source = p.split('_')
        if int(source) + 1 > len(stats):
            stats.append({})
        stats[int(source)][par] = np.array(quantile(c, percents).tolist() + [c.std()])

    return stats


def make_row(indict, dtype, **kwargs):
    dr = deepcopy(indict)
    dr.update(kwargs)
    row = np.zeros(1, dtype=dtype)
    for k, v in dr.iteritems():
        try:
            row[k] = v
        except:
            print("could not add {} to row".format(k))
    return row

    
def get_color(chain, parnames, color=("f814w", "f160w")):
    snums = np.unique([p.split('_')[1] for p in parnames]).astype(int)
    col = np.zeros([len(snums), chain.shape[0]])
    for i, obj in enumerate(snums):
        ind0 = parnames.tolist().index("{}_{:.0f}".format(color[0], obj))
        ind1 = parnames.tolist().index("{}_{:.0f}".format(color[1], obj))
        col[i,:] = -2.5 * np.log10(chain[:, ind0] / chain[:, ind1])

    return col, snums


def get_cat_dtype():
    fitcols = ["ra", "dec", "f814w", "f160w", "q", "pa", "n", "r"]
    coldesc = [("id", "S20"), ("region", np.int), ("cmean", np.float), ("csig", np.float)]
    coldesc += [(c, (np.float, 4)) for c in fitcols]
    return np.dtype(coldesc)


def match(c1, c2, dmatch=0.5):
    c1_idx, sep, _ = c2.match_to_catalog_sky(c1)
    # idx = inds of c1 that match to c2
    good = sep.arcsec < dmatch
    return c1_idx[good], np.arange(len(c2))[good], sep[good]


if __name__ == "__main__":

    vers = "v0"
    names = glob.glob("../odyresults/{}/*pkl".format(vers))
    savedir = "../odyresults/{}/figures/".format(vers)
    savedir = ''

    twall, ncall, npix = [], [], []
    nsource, regnum, rows = [], [], []
    ctype = get_cat_dtype()

    #sys.exit()

    for n in names:
        print(n)
        result = display(n, savedir=savedir, show=False, scale_model=True, scale_residuals=True)
        cc = n.split('_')
        reg = int([c for c in cc if "reg" in c][0].replace("reg", ""))

        twall.append(result.wall_time)
        ncall.append(result.ncall)
        npix.append(np.min([s.npix for s in result.stamps]))
        nsource.append(len(result.scene.sources))
        regnum.append(reg)
        
        ids = ["{}_{}".format(reg, s.id) for s in result.scene.sources]
        stats = get_scene_stats(result.chain, result.scene.parameter_names)
        colors, ns = get_color(result.chain, result.scene.parameter_names)
        cmean, csig = colors.mean(axis=-1), colors.std(axis=-1)
        rows.append([make_row(stat, ctype, id=i, region=reg, csig=cs, cmean=cm)
                     for stat, i, cs, cm in zip(stats, ids, csig, cmean)])


    cat = np.concatenate(rows)
    fits.writeto("xdf_forcecat_{}.fits".format(vers), cat, overwrite=True)

    twall = np.array(twall)
    ncall = np.array(ncall)
    npix = np.array(npix)
    nsource = np.array(nsource)
    regnum = np.array(regnum)

    threedhst_cat = np.genfromtxt('../data/catalogs/3DHST_combined_catalog.dat', names=True)
    hst = SkyCoord(ra=threedhst_cat['ra'], dec=threedhst_cat['dec'], unit=(u.deg, u.deg))
    force = SkyCoord(ra=cat['ra'][:, 0, 0], dec=cat['dec'][:, 0, 0], unit=(u.deg, u.deg))
    ih, ix, sep = match(hst, force)

    mag3d = threedhst_cat["mag_H"]
    magf = 25.94 - 2.5 * np.log10(cat["f160w"][:, 0, 0])
    bands = "f814w", "f160w"
    msig = np.array([1.086 * cat[b][: , 0, -1] / cat[b][: , 0, 0] for b in bands])
    csig_marg = np.hypot(*msig)

    mag3d = threedhst_cat["mag_H"]
    magf = 25.94 - 2.5 * np.log10(cat["f160w"][:, 0, 0])
    col3d = threedhst_cat["mag_I"] - threedhst_cat["mag_H"]
    
    idxc, idxcatalog, d2d, d3d = force.search_around_sky(force[ix], 0.6*u.arcsec) 
    ids, counts = np.unique(idxc, return_counts=True)
    iso = ids[counts == 1]

    ms = 6
    
    hfig, hax = pl.subplots()
    hax.plot(np.linspace(22, 30, 100), np.zeros(100), 'k:', linewidth=3)
    hax.errorbar(mag3d[ih], mag3d[ih] - magf[ix], yerr=msig[1][ix],
                 linestyle='',marker='o', markersize=ms, label="all")
    hax.errorbar(mag3d[ih][iso], (mag3d[ih] - magf[ix])[iso], yerr=msig[1][ix][iso],
                 linestyle='', color='tomato', marker='o', markersize=ms, label="isolated")
    hax.set_xlabel("m$_{f160w}$ (3DHST)")
    hax.set_ylabel("$\Delta m_{f160w}$ (3DHST - ForceXDF)")
    hax.legend()
    hax.set_xlim(22.2, 27.8)
    hfig.savefig("figures/mag_compare.pdf")


    cfig, cax = pl.subplots()
    cax.errorbar(col3d[ih], cat["cmean"][ix], yerr=cat["csig"][ix], linestyle="", marker='')
    cax.plot(col3d[ih][iso], cat["cmean"][ix][iso, 0], 'o', color="tomato")
    cax.plot(np.linspace(-1, 3, 100), np.linspace(-1, 3, 100), 'k:')
    cax.set_xlabel("I-H (3DHST)")
    cax.set_ylabel("I-H (Forcepho)")
    cfig.savefig("figures/color_compare.pdf")
    
    
    efig, eax = pl.subplots()
    ib = 1
    eax.plot(magf, msig[ib], 'o', markersize=ms)
    eax.set_xlabel('m$_{{{}}}$'.format(bands[ib]))
    eax.set_ylabel('$\sigma_{{{}}}$'.format(bands[ib]))
    eax.set_ylim(0, 0.5)
    efig.savefig("figures/mag_errors.pdf")

    cfig, cax = pl.subplots()
    cax.plot(cat["csig"], csig_marg, 'o', markersize=ms)
    x = np.linspace(0, 1.0, 20)
    cax.plot(x, x, 'k:', linewidth=3)
    cax.set_xlim(0, 0.5)
    cax.set_ylim(0, 0.5)
    cax.set_xlabel("$\sigma_{{({} - {})}}$".format(*bands))
    cax.set_ylabel("$\sqrt{{\sigma_{{{}}}^2 + \sigma_{{{}}}^2}}$".format(*bands))
    cfig.savefig("figures/color_errors.pdf")
