import os, sys
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.patches import Rectangle

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


def match(c1, c2, dmatch=0.5):
    c1_idx, sep, _ = c2.match_to_catalog_sky(c1)
    # idx = inds of c1 that match to c2
    good = sep.arcsec < dmatch
    return c1_idx[good], np.arange(len(c2))[good]

def in_region(xx, yy, corners):
    in_this = ((xx > corners[0]) & (xx < corners[1]) &
               (yy > corners[2]) & (yy < corners[3]))

    return in_this
    
    
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
    mmse_cat = np.array(fits.getdata(catname))
    mmse = SkyCoord(ra=mmse_cat['ra'], dec=mmse_cat['dec'], unit=(u.deg, u.deg))
    

    threedhst_cat = np.genfromtxt(os.path.join(base, '../catalogs/3DHST_combined_catalog.dat'), names=True)
    hst = SkyCoord(ra=threedhst_cat['ra'], dec=threedhst_cat['dec'], unit=(u.deg, u.deg))
    # origin: coordinate in the upper left corner of the image, for Numpy should be 0
    xp, yp = hst.to_pixel(wcs, origin=0)
    hst_original = np.array([xp, yp])

    # only required for checking positions
    minds, hinds = match(mmse, hst)
    
    in_a_stamp = np.zeros(len(mmse_cat)) - 1

    fluxorder = np.argsort(mmse_cat["flux"])
    regions = []
    buffersize = 2

    # --- define regions and assig sources to them
    for i, o in enumerate(fluxorder):
        if in_a_stamp[o] >= 0:
            continue

        row = mmse_cat[o]
        x, y = row["x"], row["y"] 

        dx = dy = min(max(row["a"] * 10., 16.), 64.) / 2.0
        corners = np.array([x - dx, x + dx, y-dy, y+dy])
        in_this = in_region(mmse_cat["x"], mmse_cat["y"], corners)

        print(len(regions), in_this.sum())
        #in_a_stamp[in_this & (in_a_stamp < 0)] = len(regions)
        in_a_stamp[in_this] = len(regions)
        in_a_stamp[o] = len(regions)

        dx += buffersize
        dy += buffersize
        corners = np.array([x - dx, x + dx, y-dy, y+dy])
        corners = np.clip(np.round(corners), 0, mmse_size[0])
        regions.append(tuple(corners.astype(int)))

    # --- Plot the image ---
    fig, ax = pl.subplots()
    # Note we don't transpose because we haven't transposed from the fits input
    cm = ax.imshow(im, origin="lower")
    cbar = pl.colorbar(cm, ax=ax)
    ax.plot(mmse_cat["x"], mmse_cat["y"], 'o')
    hst_idx = (hst_original.min(axis=0) > 0) & (hst_original.max(axis=0) < 500)
    ax.plot(*hst_original[:, hst_idx], marker='o', linestyle='')

    # --- clean up the regions and get stats ---
    rinds, _ = np.unique(in_a_stamp, return_counts=True)
    reg = [regions[int(i)] for i in rinds]
    regions = reg
    ngal = np.array([in_region(mmse_cat["x"], mmse_cat["y"], c).sum() for c in regions])
    sizes = np.array([c[1] - c[0] for c in regions])

    # --- write and plot the regions ---
    colors = ["tomato"]
    rfile = open("xdf_regions.dat", "w")
    rfile.write("xlo   xhi   ylo   yhi    npix  nsource\n")
    line = "{:3.0f}  {:3.0f}  {:3.0f}  {:3.0f}  {:4.0f}  {:2.0f}\n"
    for i,corners in enumerate(regions):
        xy = (corners[0], corners[2])
        r = Rectangle(xy, corners[1] - corners[0], corners[3] - corners[2],
                      alpha=0.2, color=colors[np.mod(i, len(colors))])
        ax.add_patch(r)
        ax.text(xy[0], xy[1], "{:3.0f}:{:2.0f}".format(i, ngal[i]))
        vals = list(corners) + [sizes[i]**2, ngal[i]]
        rfile.write(line.format(*vals))

    rfile.close()    
    pl.show()
    sys.exit()

    
    CANDELS_Cat = Vizier.get_catalogs(catalog="J/ApJS/207/24")[0]
    # just keep galaxies and good objects
    idx_good_sources = (CANDELS_Cat['Q'] == 0.0)
    idx_galaxy_sources = (CANDELS_Cat['S_G'] < 0.95)
    CANDELS_Cat = CANDELS_Cat[idx_good_sources]
    guo = SkyCoord(ra=CANDELS_Cat['RAJ2000'], dec=CANDELS_Cat['DEJ2000'], unit=(u.hourangle, u.deg))
    # origin: coordinate in the upper left corner of the image, for Numpy should be 0
    xp, yp = guo.to_pixel(wcs, origin=0)
    guo_original = np.array([xp, yp])
