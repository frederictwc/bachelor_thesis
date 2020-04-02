import numpy as np
import glob, os
import re
import scipy
import scipy.interpolate
number_of_cells = 100
from numpy import array,flipud


def output_text_file():
     t = []
     u = []
     v = []
     for filenames in sorted(glob.glob("*.vtk")):
    
         
            d = read_vtk(filenames) 
            ot = interp_to_uniform(d["x"], d["y"], fields=[d["T"]], n=number_of_cells)
            ou = interp_to_uniform(d["x"], d["y"], fields=[d["U"]], n=number_of_cells)
            ov = interp_to_uniform(d["x"], d["y"], fields=[d["V"]], n=number_of_cells)
            rt = ot[2]
            pt = rt[0]
            ru = ou[2]
            pu = ru[0]
            rv = ov[2]
            pv = rv[0]
            
            pt= np.reshape(flipud(pt),(number_of_cells**2))
            t.append(pt)
            pu= np.reshape(flipud(pu),(number_of_cells**2))
            u.append(pu)
            pv= np.reshape(flipud(pv),(number_of_cells**2))
            v.append(pv)    
          
     t2 = np.asarray(t)
     t3 = np.reshape(t2,t2.size)
     u2 = np.asarray(u)
     u3 = np.reshape(u2,u2.size)
     v2 = np.asarray(v)
     v3 = np.reshape(v2,v2.size)


     np.savetxt('t_raw.out', t3)
     np.savetxt('u_raw.out', u3)
     np.savetxt('v_raw.out', v3)

def read_vtk(fn):
    s = 0    # initial state
    n = None # num points
    nc = None # num cells
    vn = None # current variable name
    d = dict()
    i = None # current array index
    x = None; y = None; z = None;  # point arrays
    a = None # current array
    cc = None  # cells
    ct = None  # cell types
    m = None # temporary buffer
    with open(fn) as f:
        for l in f:
            if s == 0: # parse header
                if l.count("POINTS"):
                    n = int(re.findall("\D*(\d*)\D*", l)[0])
                    assert n > 0
                    i = 0
                    x = np.empty(n); y = np.empty(n); z = np.empty(n)
                    s = 10
                elif l.count("POINT_DATA"):
                    nd = int(re.findall("\D*(\d*)", l)[0])
                    assert nd == n
                    s = 20
                elif l.count("CELLS"):
                    sp = l.split()
                    m = int(sp[1])
                    nc = m if nc is None else nc
                    assert nc == m
                    assert nc > 0
                    sz = int(sp[2])
                    assert (sz - nc) % nc == 0
                    shape = (nc, (sz - nc) // nc) # assume same size
                    cc = np.empty(shape, dtype=int)
                    i = 0
                    s = 30
                elif l.count("CELL_TYPES"):
                    m = int(l.split()[1])
                    nc = m if nc is None else nc
                    assert nc == m
                    assert nc > 0
                    ct = np.empty(nc, dtype=int)
                    i = 0
                    s = 40
            elif s == 10: # read point coordinates
                sp = l.split()
                if len(sp) == 0:
                    continue
                assert len(sp) == 3
                x[i], y[i], z[i] = map(float, sp)
                i += 1
                if i >= n:
                    d['x'] = x; d['y'] = y; d['z'] = z
                    s = 0
            elif s == 20: # read points header
                sp = l.split()
                if len(sp) == 0:
                    continue
                assert sp[0] == "SCALARS"
                assert sp[2] == "float"
                vn = sp[1]
                a = np.empty(n)
                i = 0
                s = 21
            elif s == 21: # skip next line
                sp = l.split()
                assert sp[0] == "LOOKUP_TABLE"
                assert sp[1] == "default"
                s = 22
            elif s == 22: # read point data
                a[i] = float(l)
                i += 1
                if i >= n:
                    d[vn] = a
                    s = 20
            elif s == 30: # read cell point lists
                sp = l.split()
                if len(sp) == 0:
                    continue
                assert len(sp) == int(sp[0]) + 1
                cc[i] = np.array(sp[1:], dtype=int)
                i += 1
                if i >= nc:
                    d['cells'] = cc
                    s = 0
            elif s == 40: # read cell types
                sp = l.split()
                if len(sp) == 0:
                    continue
                assert len(sp) == 1
                ct[i] = int(sp[0])
                i += 1
                if i >= nc:
                    d['cell_types'] = ct
                    s = 0
    return d

def get_cell_volume(x, y, z, cc, ct):
    # check same type
    ct0 = ct[0]
    assert np.all(ct == ct0)

    # calc cell volume, assume cubes
    cv = None
    if ct0 == 8: # VTK pixel
        assert(cc.shape[1] == 4)
        cv = (x[cc[:,1]] - x[cc[:,0]]) * (y[cc[:,2]] - y[cc[:,0]])
    elif ct0 == 11: # VTK voxel
        assert(cc.shape[1] == 8)
        cv = (x[cc[:,1]] - x[cc[:,0]]) * (y[cc[:,2]] - y[cc[:,0]]) * (z[cc[:,4]] - y[cc[:,0]])
    assert np.all(cv > 0.)
    return cv

def interp_to_cells(d):
    cc = d["cells"]
    ct = d["cell_types"]
    nc = len(cc)

    dc = dict()

    # calc averages
    excl = ["cells", "cell_types"]
    for fk in d.keys():
        if fk in excl:
            continue
        dc[fk] = np.mean(d[fk][cc], axis=1)

    # calc cell volume
    dc['cv'] = get_cell_volume(d['x'], d['y'], d['z'], cc, ct)
    assert 'cv' not in d.keys()

    return dc


# Calc weights of points by size of neighbouring cells
def get_weights(d):
    cc = d["cells"]
    ct = d["cell_types"]
    nc = len(cc)

    cv = get_cell_volume(d['x'], d['y'], d['z'], cc, ct)

    w = np.zeros_like(d['x'])
    csz = cc.shape[1]  # points per cell
    for i in range(csz):
        np.add.at(w, cc[:,i], cv / cc.shape[1])
    assert np.isclose(cv.sum(), w.sum())
    return w


# result follows get_series()
def get_series_vtk(pathdir, interpcell=False):
    ff = sorted(glob.glob(os.path.join(pathdir, "*.vtk")))

    alias = {'x':'x', 'y':'y', 'z':'z',
            'U':'u', 'V':'v', 'W':'w',
            'P':'p', 'T':'f', 'cv':'cv'}

    dd = dict()

    for f in ff:
        d = read_vtk(f)
        if interpcell:
            d = interp_to_cells(d)
        else:
            d['cv'] = get_weights(d)

        if not dd:
            # init with empty lists
            for k in d.keys():
                a = alias[k] if k in alias.keys() else k
                assert a not in dd.keys()
                dd[a] = []

        for k in d.keys():
            a = alias[k] if k in alias.keys() else k
            dd[a].append(d[k])

    return dd

def interp_to_uniform(xp, yp, fields=[], n=None):
    if n is None:
        n = int(len(xp) ** 0.5 + 0.5)

    x1 = np.linspace(xp.min(), xp.max(), n)
    y1 = np.linspace(yp.min(), yp.max() - 1e-6 * yp.ptp(), n)
        # ADHOC: avoid fill_value in interpolate

    x, y = np.meshgrid(x1, y1)
    pts = np.vstack((xp, yp)).T

    r = []
    for fp in fields:
        f = scipy.interpolate.griddata(pts, fp, (x, y), method="cubic")
        assert np.all(np.isfinite(f))
        r.append(f)

    return x1, y1, r


output_text_file()
