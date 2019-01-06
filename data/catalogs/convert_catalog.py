# This script converts the MMSE catalog generated from a curour of the 60mas
# XDF images into a fits binary table with RA/Dec values

import numpy as np
import json
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

catname = "/Users/bjohnson/Projects/xdf/data/catalogs/xdf_f160-f814_3020-3470.pkl"
imname = "/Users/bjohnson/Projects/xdf/data/images/hlsp_xdf_hst_wfc3ir-60mas_hudf_f160w_v1_sci.fits"

mmse_position = (3020, 3470)
mmse_size = (500, 500)

sci = fits.open(imname)
wcs = WCS(sci[0].header)
cutout_image = Cutout2D(sci[0].data, mmse_position, mmse_size, wcs=wcs)
cutout_wcs = cutout_image.wcs

#import pickle
#with open(catname, "rb") as f:
#    cat = pickle.load(f)
#dat = np.array(cat.to_records())

with open(catname.replace(".pkl", ".json"), "r") as f:
    dd = json.load(f)

cols = dd.keys()
cols = [("x", np.float), ("y", np.float), ("ra", np.float), ("dec", np.float),
        ("flux", np.float),
        ("a", np.float), ("b", np.float), ("theta", np.float),
        ("cxx", np.float), ("cyy", np.float), ("cxy",np.float),
        ("band", "S6"), ("mode", "S4"), ("id", "S32")]
dt = np.dtype(cols)
cn = dt.names
nkeep = (np.array(dd["keep"].values()) == True).sum()
kk = dd[cn[0]].keys()
    
data = np.zeros(nkeep, dtype=dt)
j = 0
for i,k in enumerate(kk):
    if dd["keep"][k] != True:
        continue
    data[j]["id"] = k
    for c in cn[:-1]:
        if c not in dd.keys():
            continue
        data[j][c] = dd[c][k]
    j += 1

ra, dec = cutout_wcs.wcs_pix2world(data["x"], data["y"], 1)
data["ra"] = ra
data["dec"] = dec

fits.writeto(catname.replace(".pkl", ".fits"), data, overwrite=True)

#with open(catname.replace(".pkl", ".npz"), "wb") as f:
#    np.savez(f, mmse_xdf=dat)

#cols = [a for a in cat.columns]
#dt = [cat[c].dtype for c in cols]
#for d in dt:
#    if d == np.dtype('bool'):
        
#dtype = np.dtype([(n, d) for n, d in zip(cols, dt)])
#dat = [np.array(cat[n], dtype=d) for n, d in zip(cols, dt)]
