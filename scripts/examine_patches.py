import numpy as np
import glob, os, sys
from os.path import join as pjoin
import h5py

udf_dir = "/Users/bjohnson/Projects/xdf/"

if __name__ == "__main__":
    
    path_to_patches = "data/patches/20190612_9x9_threedhst/"
    
    patches = glob.glob(pjoin(udf_dir, path_to_patches, "*h5"))
    
    for patch in patches:
        with h5py.File(patch, "r") as hfile:
            pars = hfile["mini_scene"]["sourcepars"][:]
            nsource = len(pars)
            filter_name = hfile['images'].attrs['filters'][0]
            exp_name = hfile['images'][filter_name].attrs['exposures'][0]
            data = hfile['images'][filter_name][exp_name]
            ierr = 1.0 / np.array(data['rms']).T
            mask = np.array(data['mask']).T
            bad = ~np.isfinite(ierr) | (mask == 1.0) | (ierr == 0)
            nbad = np.sum(bad)
            
            print(os.path.basename(patch), nsource, nbad)