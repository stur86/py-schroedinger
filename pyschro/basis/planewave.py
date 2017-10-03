# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.constants as cnst
import scipy.sparse as scisp
import scipy.sparse.linalg as scisp_lin

from pyschro.grid import Grid
from pyschro.basis.basis import BasisSet
from pyschro.utils import multikron

class PlanewaveBasis(BasisSet):

    def __init__(self, grid, V, m):

        super(PlanewaveBasis, self).__init__(grid, V, m)

        # Now define the k grid and the Hamiltonian
        kbounds = []
        for i, s in enumerate(grid.size):
            if s%2 == 0:
                kbounds.append((-s/2.0/(grid.dx[i]*s),
                                (s/2.0-1)/(grid.dx[i]*s)))
            else:
                kbounds.append((-(s-1)/2.0/(grid.dx[i]*s),
                                (s-1)/2.0/(grid.dx[i]*s)))

        self.kgrid = Grid(kbounds, grid.size)
        self.repr = np.exp(-1.0j*2.0*np.pi*\
                           np.sum(self.kgrid.grid_lin[:,None,:]*\
                                  self.spacegrid.grid_lin[None,:,:],
                                  axis=-1))
        self.repr /= np.sqrt(np.prod(self.kgrid.size))

        # And the potential
        self.H = np.dot(self.repr, np.dot(np.diag(self.V),
                                          np.conj(self.repr.T)))

        # Now for the kinetic component. This is easy, as a wavefunction of
        # form exp(-i*k*x) has second derivative -k**2 * exp(-i*k*x).
        # So that's all we need.
        self.H += (cnst.hbar**2.0)/(2.0*m)*\
                  (np.diag(4.0*np.pi**2.0*\
                   np.linalg.norm(self.kgrid.grid_lin, axis=-1)**2))        

