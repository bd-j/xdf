# coding: utf-8

#!/usr/bin/env python2

"""
Created on November 29, 2018

@author: sandro.tacchella@cfa.harvard.edu

Comments:
- need to sort out good/bad PSF before stacking
- background subtraction
"""

# import modules

import numpy as np
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt
from scipy import ndimage
from astropy.modeling import models, fitting

from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1


# define functions

def prepare_image(path_image, show_img=False):
    '''
    Loads image, determines center, readius and WCS of image.
    '''
    img_fits = fits.open(path_image)
    # determine center of image
    wcs_img = WCS(img_fits[0].header)
    coord_center = wcs_img.wcs_pix2world([[int(0.5*img_fits[0].header['NAXIS1']), int(0.5*img_fits[0].header['NAXIS2'])]], 1)
    # determine radius of image
    coord_edge = wcs_img.wcs_pix2world([[0, 0]], 1)
    radius_arcsec = np.sqrt((coord_edge[0][0]-coord_center[0][0])**2+(coord_edge[0][1]-coord_center[0][1])**2)*3600
    # show image
    if show_img:
        # plot image
        fig = plt.figure()
        fig.add_subplot(111, projection=wcs_img)
        plt.imshow(img_fits[0].data, origin='lower', cmap=plt.cm.viridis, vmin=0.0, vmax=0.05)
        plt.plot(int(0.5*img_fits[0].header['NAXIS1']), int(0.5*img_fits[0].header['NAXIS2']), '+', color='red')
        plt.xlabel('RA')
        plt.ylabel('Dec')
        plt.show()
    return(img_fits[0].data, coord_center, radius_arcsec, wcs_img)


def get_gaia_star(coord_center, radius_arcsec, g_mag_cut=15.0):
    '''
    Retruns list of coordinates of GAIA stars in the field.
    '''
    # set up coordinates
    coord = SkyCoord(ra=coord_center[0][0], dec=coord_center[0][1], unit=(u.degree, u.degree), frame='icrs')
    radius = u.Quantity(radius_arcsec, u.arcsec)
    # query GAIA stars
    gaia_stars = Gaia.query_object_async(coordinate=coord, radius=radius)
    # choose stars with proper motion and mag cut
    prop_motion = np.sqrt(gaia_stars['pmra'].data.data**2+gaia_stars['pmdec'].data.data**2)
    idx_good_stars = (gaia_stars['astrometric_excess_noise'].data.data < 1.0) & (prop_motion < 20.0).data & (gaia_stars['phot_g_mean_mag'].data.data > 15.0)
    print('number of stars found in GAIA :', np.sum(idx_good_stars))
    coord_gaia = SkyCoord(ra=gaia_stars[idx_good_stars]['ra'], dec=gaia_stars[idx_good_stars]['dec'], unit=(u.deg, u.deg))
    return(coord_gaia)

    
def make_cutout(image, WCS_img, position, size=100.0, size_in_arcsec=False, show_img=False):
    '''
    Makes cutout.
    '''
    # get coordinate frame
    WCS_img.sip = None
    # define position
    if size_in_arcsec:
        size = u.Quantity((size, size), u.arcsec)
        cutout = Cutout2D(image, position, size, wcs=WCS_img)
    else:
        size = (size*u.pixel, size*u.pixel)
        cutout = Cutout2D(image, WCS_img.wcs_world2pix([[position.ra.deg, position.dec.deg]], 1)[0], size)
    if show_img:
        img = plt.imshow(cutout.data, origin='lower')
        plt.axis('off')
        plt.show()
    return(cutout)


def fit_star(psf_data, type_fit):
    '''
    Fits Gaussian or Moffat model to star.
    '''
    # Fit the data using astropy.modeling
    if (type_fit == 'Gaussian'):
        p_init = models.Gaussian2D(amplitude=np.nanmax(psf_data), x_mean=0.5*psf_data.shape[0], y_mean=0.5*psf_data.shape[1])
    elif (type_fit == 'Moffat'):
        p_init = models.Moffat2D(amplitude=np.nanmax(psf_data), x_0=0.5*psf_data.shape[0], y_0=0.5*psf_data.shape[1])
    fit_p = fitting.LevMarLSQFitter()
    x, y = np.mgrid[:psf_data.shape[0], :psf_data.shape[1]]
    p = fit_p(p_init, x, y, psf_data)
    return(p)


def get_PSF(path_image, path_psf, psf_size, factor_enlarge=1.2, g_mag_cut=15.0, verbose=False):
    '''
    Determines PSF from image from GAIA stars.
    '''
    # load image
    img_data, coord_center, radius_arcsec, wcs_img = prepare_image(path_image, show_img=False)
    # laod GAIA stars
    coord_gaia = get_gaia_star(coord_center, radius_arcsec, g_mag_cut=g_mag_cut)
    # iterate over all stars in GAIA
    psf_mat = np.nan*np.zeros([len(coord_gaia)/2, int(factor_enlarge*psf_size), int(factor_enlarge*psf_size)])
    for ii_star in range(len(coord_gaia)/2):
        star_cutout = make_cutout(img_data, wcs_img, coord_gaia[ii_star], size=factor_enlarge*psf_size, show_img=verbose)
        star_fit = fit_star(star_cutout.data, type_fit='Moffat')
        # print star_fit
        dx, dy = 0.5*star_cutout.data.shape[0]-star_fit.parameters[1], 0.5*star_cutout.data.shape[1]-star_fit.parameters[2]
        if verbose:
            print('shift dx, dy = ', dx, dy)
        # shift PSF
        star_cutout_shifted = ndimage.interpolation.shift(star_cutout.data, [dx, dy], order=3)
        star_cutout_shifted_norm = star_cutout_shifted/np.nansum(star_cutout_shifted)
        psf_mat[ii_star, :, :] = star_cutout_shifted_norm
    # combine PSFs to one PSF
    PSF_combined = np.median(psf_mat, axis=0)
    # cut to right size
    edge_cut = int(0.5*(psf_mat.shape[1]-psf_size))
    PSF_final = PSF_combined[edge_cut:-edge_cut, edge_cut:-edge_cut]
    # save and return PSF
    hdu = fits.PrimaryHDU(data=PSF_final)
    hdu.writeto(path_psf, overwrite=True)
    return(PSF_final)


# run example:
psf_base = "/Users/sandrotacchella/ASTRO/JWST/xdf/data/psfs/"

path_image = 'https://archive.stsci.edu/pub/hlsp/xdf/hlsp_xdf_hst_acswfc-60mas_hudf_f814w_v1_sci.fits'
path_PSF = psf_base + 'PSF_f814w.fits'
PSF = get_PSF(path_image, path_PSF, 100.0, factor_enlarge=1.2, g_mag_cut=15.0, verbose=False)
plt.imshow(np.log10(PSF))
plt.show()

path_image = 'https://archive.stsci.edu/pub/hlsp/xdf/hlsp_xdf_hst_wfc3ir-60mas_hudf_f125w_v1_sci.fits'
path_PSF = psf_base + 'PSF_f125w.fits'
PSF = get_PSF(path_image, path_PSF, 100.0, factor_enlarge=1.2, g_mag_cut=15.0, verbose=False)
plt.imshow(np.log10(PSF))
plt.show()

path_image = 'https://archive.stsci.edu/pub/hlsp/xdf/hlsp_xdf_hst_wfc3ir-60mas_hudf_f140w_v1_sci.fits'
path_PSF = psf_base + 'PSF_f140w.fits'
PSF = get_PSF(path_image, path_PSF, 100.0, factor_enlarge=1.2, g_mag_cut=15.0, verbose=False)
plt.imshow(np.log10(PSF))
plt.show()

path_image = 'https://archive.stsci.edu/pub/hlsp/xdf/hlsp_xdf_hst_wfc3ir-60mas_hudf_f160w_v1_sci.fits'
path_PSF = psf_base + 'PSF_f160w.fits'
PSF = get_PSF(path_image, path_PSF, 100.0, factor_enlarge=1.2, g_mag_cut=15.0, verbose=False)
plt.imshow(np.log10(PSF))
plt.show()



