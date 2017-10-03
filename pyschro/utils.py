"""Random utility functions"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import scipy.sparse as scisp

# A utility: chain kronecker product
def multikron(a):
    if len(a) == 1:
        return a[0]
    elif len(a) == 2:
        return scisp.kron(*a)
    elif len(a) > 2:
        return scisp.kron(a[0], multikron(a[1:]))

# Mutually exclusive arguments - only one can be present!
def only_one(*args):

    return sum([x is None for x in args]) == 1

# On which side of a given plane are we?
def plane_side(p, p0, n):
    return np.sign(np.tensordot((np.array(p, copy=False)-p0),
                                 n, axes=(-1,-1)))

def savetxt_miw(fname, miw_hist,
                traj_dim=None,
                E_ref=None,
                gnuplot_file=False,
                grid=None):

    # Save a TXT file containing a MIW history
    Earr = miw_hist.E
    if E_ref is not None:
        Earr = (Earr-E_ref)/E_ref

    data = np.array([miw_hist.t, Earr, miw_hist.T]).T

    if traj_dim is not None:
        # 2D or 3D trajectories?
        is_3d = False
        try:
            if len(traj_dim) == 2:
                is_3d = True
            else:
                raise ValueError("Invalid traj_dim")
        except TypeError:
            traj_dim = [traj_dim]
        data = np.concatenate((data, miw_hist.w_x[:,:,traj_dim[0]]), axis=1)
        if is_3d:
            data = np.concatenate((data, miw_hist.w_x[:,:,traj_dim[1]]),
                                   axis=1)

    np.savetxt(fname, data)

    # If gnuplot file is True, create a file for the trajectories
    if traj_dim is not None and gnuplot_file:
        gpfname = os.path.splitext(fname)[0] + '.gp'
        gpf = open(gpfname, 'w')
        gpf.write('set xtics nomirror\nset ytics nomirror\n')
        gpf.write('set palette model HSV defined (0 0 1 0.8, 1 0.85 1 0.8)\n')
        if grid is not None:
            gpf.write('set xrange [{0}:{1}]\n'.format(*grid.bounds[traj_dim[0]]))
            if is_3d:
                gpf.write('set yrange [{0}:{1}]\n'.format(*grid.bounds[traj_dim[1]]))
        if not is_3d:
            gpf.write('plot "{0}" '.format(fname))
            gpf.write(', "" '.join([' u 1:{0}:({1}) w l lc palette notitle'.format(i+4, i)
                                 for i in range(miw_hist.w_x.shape[1])]))
        else:
            gpf.write('splot "{0}" '.format(fname))
            gpf.write(', "" '.join([' u {0}:{1}:1:({2}) w l lc palette notitle'.format(i+4,
                                                                      i+4+miw_hist.w_x.shape[1],
                                                                      i)
                                 for i in range(miw_hist.w_x.shape[1])]))            
        gpf.write('\npause -1\n')

def savetxt_tunnhist(fname, hist, p0=None, n=None, r=None, grid=None,
                     force_norm=False):

    # Save a TXT file containing a tunneling history

    tunn = []

    # Dimension?
    if grid is None:
        dim = hist.w_x[0].shape[1]
    else:
        dim = grid.dim

    if p0 is None:
        p0 = np.zeros(dim)

    if n is not None:

        n = np.array(n)

        if grid is not None:
            inds_l = grid.plane_side_indices(p0, n, -1)
            inds_r = grid.plane_side_indices(p0, n, 1)

        for i, t in enumerate(hist.t):
            if grid is not None:
                norm = 1.0 if not force_norm else sum(hist.density[i])
                frac_l = np.sum(hist.density[i][inds_l])/norm
                frac_r = np.sum(hist.density[i][inds_r])/norm
            else:
                pside = plane_side(hist.w_x[i], p0, n)
                frac_l = np.sum(pside == -1)/len(hist.w_x[i])
                frac_r = 1-frac_l
            tunn.append([t, hist.E[i], frac_l, frac_r])

    elif r is not None:

        r = r*1e-10

        if grid is not None:
            inds_in = np.where(np.linalg.norm(grid.grid_lin, axis=-1) <= r)
            inds_out = np.where(np.linalg.norm(grid.grid_lin, axis=-1) > r)

        for i, t in enumerate(hist.t):
            if grid is not None:
                norm = 1.0 if not force_norm else sum(hist.density[i])
                frac_in = np.sum(hist.density[i][inds_in])/norm
                frac_out = np.sum(hist.density[i][inds_out])/norm
            else:
                is_in = (np.linalg.norm(hist.w_x[i], axis=-1) <= r)
                frac_in = np.sum(is_in)/len(hist.w_x[i])
                frac_out = 1-frac_in
            tunn.append([t, hist.E[i], frac_in, frac_out])
    else:
        raise RuntimeError("One between n and r must be specified")

    np.savetxt(fname, np.array(tunn))