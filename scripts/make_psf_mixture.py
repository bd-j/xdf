# Script to make a psf gaussian mixture.  many things hardcoded that shouldn't be
import sys, argparse
from forcepho.mixtures import fit_psfs

# --- F160W ---
band = "hst_f160w"
# The ZC PSF image has 50mas pixels, while we want 60mas pixels for the XDF
# science images, so the PSF is oversampled by 1.2
pname, oversample = "/Users/bjohnson/Projects/xdf/data/psfs/ZC404221_H_PSF.fits", 1.2

# --- F814W 30mas ---
band = "30mas_hst_f814w"
pname, oversample = "/Users/bjohnson/Projects/xdf/data/psfs/PSF_30mas_f814w.fits", 1.0
nmix = 3
nrepeat = 4


parser = argparse.ArgumentParser()
parser.add_argument("--band", type=str,
                    default="30mas_hst_f814w",
                    help="band name")
parser.add_argument("--path_psf", type=str,
                    default="/Users/bjohnson/Projects/xdf/data/psfs/PSF_30mas_f814w.fits",
                    help="Full path to the PSF image.  Ideally has a valid WCS")
parser.add_argument("--oversample", type=float,
                    default=1.0,
                    help="factor by which the PSF image is oversampled relative to the science image." )
parser.add_argument("--ngauss", type=int,
                    default=3,
                    help="Number of gaussians in the mixtures")
parser.add_argument("--nrepeat", type=int,
                    default=4,
                    help="How many times to run the fit")


if __name__ == "__main__":

    args = parser.parse_args()
    
    ans = fit_psfs.psf_mixture(args.path_psf, args.band, nmix=args.ngauss,
                               nrepeat=args.nrepeat, oversample=args.oversample, newstyle=True)


    psfname = "gmpsf_{}_ng{}.h5".format(args.band, args.ngauss)
    from forcepho.psf import get_psf
    psf = get_psf(psfname, psf_realization=min(1, args.nrepeat))
