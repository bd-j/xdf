import os
import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.nddata import Cutout2D


from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1


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


if __name__ == "__main__":

    base = "/Users/bjohnson/Projects/xdf/data/images/"

    
    
    hband, hzp = "hlsp_xdf_hst_wfc3ir-60mas_hudf_f160w_v1_", 25.94
    iband, izp = "hlsp_xdf_hst_acswfc-60mas_hudf_f814w_v1_", 25.94
    band = hband
    mmse_position = (3020, 3470)
    mmse_size = (500, 500)
    im, wght, rms, cutout, wcs = get_cutout(base, band, mmse_position, mmse_size)
    hdr = fits.getheader(base + band + "sci.fits")

    catname = "/Users/bjohnson/Projects/xdf/data/catalogs/xdf_f160-f814_3020-3470.fits"
    mmse = np.array(fits.getdata(catname))

    threeDHST_Cat = np.genfromtxt(os.path.join(base, '../catalogs/3DHST_combined_catalog.dat'), names=True)
    hst = SkyCoord(ra=threeDHST_Cat['ra'], dec=threeDHST_Cat['dec'], unit=(u.deg, u.deg))
    # origin: coordinate in the upper left corner of the image, for Numpy should be 0
    xp, yp = hst.to_pixel(wcs, origin=0)
    hst_original = np.array([xp, yp])

    CANDELS_Cat = Vizier.get_catalogs(catalog="J/ApJS/207/24")[0]
    # just keep galaxies and good objects
    idx_good_sources = (CANDELS_Cat['Q'] == 0.0)
    idx_galaxy_sources = (CANDELS_Cat['S_G'] < 0.95)
    CANDELS_Cat = CANDELS_Cat[idx_good_sources]
    guo = SkyCoord(ra=CANDELS_Cat['RAJ2000'], dec=CANDELS_Cat['DEJ2000'], unit=(u.hourangle, u.deg))
    # origin: coordinate in the upper left corner of the image, for Numpy should be 0
    xp, yp = guo.to_pixel(wcs, origin=0)
    guo_original = np.array([xp, yp])

    
    fig, ax = pl.subplots()
    # Note we don't transpose because we haven't transposed from the fits input
    ax.imshow(im, origin="lower")
    ax.plot(mmse["x"], mmse["y"], 'o')
    hst_idx = (hst_original.min(axis=0) > 0) & (hst_original.max(axis=0) < 500)
    ax.plot(*hst_original[:, hst_idx], marker='o', linestyle='')

    xlo, xhi, ylo, yhi = 10, 40, 375, 405 #325, 345, 255, 290 #440, 480, 235, 270#148, 160, 180, 210

    hh = (hst_original[0] > xlo) & (hst_original[0] < xhi) & (hst_original[1] > ylo) & (hst_original[1] < yhi)
    print(threeDHST_Cat[hh]["mag_H"])

    pl.show()
