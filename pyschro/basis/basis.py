# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.constants as cnst


class BasisSet(object):

    def __init__(self, grid, V, m):

        # Quick check!
        if hasattr(V, '__call__'):
            V = V(grid.grid_lin)
        else:
            V = np.array(V)

        if not grid.validate(V):
            raise ValueError("Invalid vector or function V passed to Basis")

        self.spacegrid = grid
        self.V = V
        self.m = m

        # And... do nothing for now
        self.H = np.zeros((grid.linshape[0], grid.linshape[0]))

        self.repr = np.identity(grid.linshape[0])  # Basis-to-real transform

    def basis2grid(self, v):

        # Transform a vector from this basis to real space
        return np.dot(self.repr, v)

    def grid2basis(self, v):

        return np.dot(self.repr.conjugate().T, v)

    def gaussian_packet(self, x0, w, v=None):

        # Create a gaussian wavepacket on the spatial grid, then transform
        # it into the current basis and return it

        x0 = np.array(x0)
        if x0.shape != (self.spacegrid.dim,):
            raise ValueError("Invalid dimension for wavepacket center")

        gwave = np.exp(-(np.linalg.norm(self.spacegrid.grid_lin -
                                        x0, axis=1)/w)**2.0).astype(np.complex)
        gwave /= np.linalg.norm(gwave)

        # Now add the speed
        if v is not None:
            v = np.array(v)
            # Is it the right dimension?
            if v.shape != (self.spacegrid.dim,):
                # What?
                if v.shape == ():
                    v = np.array([v]*self.spacegrid.dim)
                else:
                    raise ValueError("Invalid v passed to gaussian_packet")
        else:
            v = np.zeros(x0.shape)

        gwave *= np.exp(1.0j*np.tensordot(self.spacegrid.grid_lin,
                                          v, axes=(-1, -1))*self.m/cnst.hbar)
        #gwave = np.squeeze(gwave, -1)

        return self.grid2basis(gwave)
