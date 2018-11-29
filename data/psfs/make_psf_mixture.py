# Script to make a psf gaussian mixture.  many things hardcoded that shouldn't be
import sys
from forcepho.mixtures import fit_psfs

band = "hst_f160w"
pname = "/Users/bjohnson/Projects/xdf/data/ZC404221_H_PSF.fits"
nmix = 3
nrepeat = 4
ans = fit_psfs.psf_mixture(pname, band, nmix=nmix, nrepeat=3, oversample=1.2, newstyle=True)


psfname = "gmpsf_hst_f160w_ng3.h5"
from phoutils import get_psf

psf = get_psf(psfname, psf_realization=1)
