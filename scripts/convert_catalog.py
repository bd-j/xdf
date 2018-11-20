import numpy as np
import json
import fits

catname = "/Users/bjohnson/Projects/xdf/data/xdf_f160-f814_3020-3470.pkl"
#import pickle
#with open(catname, "rb") as f:
#    cat = pickle.load(f)

#dat = np.array(cat.to_records())
    
with open(catname.replace(".pkl", ".json"), "w") as f:
    cat.to_json(f)


with open(catname.replace(".pkl", ".json"), "r") as f:
    dd = json.load(f)
    cols = dd.keys()
    cols = [("x", np.float), ("y", np.float), ("flux", np.float),
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
            data[j][c] = dd[c][k]
        j += 1


fits.writeto(catname.replace(".pkl", ".fits"), data)

#with open(catname.replace(".pkl", ".npz"), "wb") as f:
#    np.savez(f, mmse_xdf=dat)

#cols = [a for a in cat.columns]
#dt = [cat[c].dtype for c in cols]
#for d in dt:
#    if d == np.dtype('bool'):
        
#dtype = np.dtype([(n, d) for n, d in zip(cols, dt)])
#dat = [np.array(cat[n], dtype=d) for n, d in zip(cols, dt)]
