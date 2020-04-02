import matplotlib.pyplot as plt
import numpy as np
import glob, os
import re
import scipy
import scipy.interpolate
number_of_cells = 100
from numpy import array,flipud


def output_text_file():


  
     pp = sorted(glob.glob("*.vtk"))

     t = []
     u = []
     v = []
     a = 0
     i = 0
    

     number_of_cores = int(len(pp)/(len(sorted(glob.glob("*-0.vtk")))))
     number_of_time_steps = len(sorted(glob.glob("*-0.vtk")))
     print("number of cores = ",number_of_cores)
     print("number of time steps = ",number_of_time_steps)


   
     while i<number_of_time_steps:

              pp = sorted(glob.glob("*.vtk"))
              print("step:",i)
              print("file: ",sorted(glob.glob("*-0.vtk"))[i])
              pp=pp[a:a+number_of_cores-1]

              r = merge([read_vtk(p) for p in pp])

              x,y,ff = interp_to_uniform(r['x'], r['y'], ffp=[r['U'], r['V'], r['T']],n=number_of_cells-1)
                              
              pt= np.reshape(flipud(ff[2]),(number_of_cells**2))
              t.append(pt)
              pu= np.reshape(flipud(ff[0]),(number_of_cells**2))
              u.append(pu)
              pv= np.reshape(flipud(ff[1]),(number_of_cells**2))
              v.append(pv)    
                   
              t2 = np.asarray(t)
              t3 = np.reshape(t2,t2.size)
              u2 = np.asarray(u)
              u3 = np.reshape(u2,u2.size)
              v2 = np.asarray(v)
              v3 = np.reshape(v2,v2.size)
              
              #print(np.shape(t3))
              i+=1
              a += number_of_cores 

     np.savetxt('t_raw.out', t3)
     np.savetxt('u_raw.out', u3) 
     np.savetxt('v_raw.out', v3)         

   







     
# Reads VTK unstructured grid with point data.
# fn: filename
# Returns:
# dict() with fields:
#   x,y,z: points
#   cells: lists of neighbour points for each cell
#   cell_types: type of each cell
#   ... point fields found in fn (e.g. 'T', 'U', 'V')
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


# Merges multiple VTK unstructured grids into one
# XXX: ignoring cells.
# vv: list of outputs from read_vtk() for multiple files
# Returns:
# merged grid in format of read_vtk()
def merge(vv):
    if not vv:
        return None

    r = dict()

    v0 = vv[0]

    ff = v0.keys()
    ff = [f for f in ff if f not in ["cells", "cell_types"]]

    for v in vv:
        for f in ff:
            if f not in r: r[f] = np.array([])
            r[f] = np.append(r[f], v[f])
    return r


# Interpolates point data to uniform grid.
# xp, yp: points, shape=(n)
# ffp: lists of arrays of shape (n)
# n: number of cells in uniform mesh: nx=ny=n;
#    if None, detect nx,ny from xp,yp
# Returns:
# x1,y1: 1d coordinates of points, shape=(nx+1),(ny+1)
# r: list of 2d arrays,
def interp_to_uniform(xp, yp, ffp=[], n=None):
    if n is None:
        hx = np.unique(np.abs(xp - np.roll(xp, 1)))[1]
        hy = np.unique(np.abs(yp - np.roll(yp, 1)))[1]
        lx = xp.ptp()
        ly = yp.ptp()
        nx = int(lx / hx + 0.5) # number of cells
        ny = int(ly / hy + 0.5)
    else:
        nx = n
        ny = n

    x1 = np.linspace(xp.min(), xp.max(), nx + 1)
    y1 = np.linspace(yp.min(), yp.max(), ny + 1)

    x, y = np.meshgrid(x1, y1)
    xyp = np.vstack((xp, yp)).T

    ff = []
    for fp in ffp:
        f = scipy.interpolate.griddata(xyp, fp, (x, y),
            method="cubic", fill_value=0.)
        ff.append(f)

    return x1, y1, ff



output_text_file()



