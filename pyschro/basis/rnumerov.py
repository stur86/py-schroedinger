# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.constants as cnst
import scipy.sparse as scisp
import scipy.sparse.linalg as scisp_lin
import scipy.linalg as sci_lin

from pyschro.basis.basis import BasisSet
from pyschro.utils import multikron


class Radial3DNumerovBasis(BasisSet):

    def __init__(self, grid, V, m, l=0):

        super(Radial3DNumerovBasis, self).__init__(grid, V, m)

        # Let's create the Hamiltonian!
        #
        # This changes depending on whether we're using a logarithmic grid or
        # not. Generally, you'd like to use it, but if not, we've got you
        # covered too.

        if grid.dim > 1:
            raise ValueError('Grid for radial 3D Numerov basis set must have'
                             ' only one dimension.')

        gn = grid.size[0]
        D2 = scisp.spdiags([[1.0]*gn, [-2.0]*gn, [1.0]*gn],
                           (-1, 0, 1), gn, gn, format='csc')

        # Now we build the full kinetic operator
        dx = np.log(grid.dx[0])
        R2 = scisp.spdiags([grid.grid_lin[:, 0]**2],
                           (0,), gn, gn, format='csc')
        A = D2/dx**2.0
        B = scisp.identity(gn, format='csc')
        B += D2/12.0
        K = -(cnst.hbar**2.0)/(2.0*m)*(scisp_lin.inv(B)*A)

        # And here's the Hamiltonian

        self._K = K
        self._V = scisp.spdiags((self.V), [0], gn, gn, format='csc')
        self._VR2 = scisp.spdiags((self.V*grid.grid_lin[:, 0]**2,), [0],
                                  gn, gn,
                                  format='csc')
        self._R2 = R2

        self.set_momentum(l)

    def set_momentum(self, l):

        gn = self.spacegrid.size[0]

        self.l = l
        if self.spacegrid.log:
            self._Vrot = ((cnst.hbar**2.0)/(2.0*self.m)*(l+0.5)**2 *
                          scisp.identity(gn, format='csc'))
            self.H = self._K + self._VR2 + self._Vrot
        else:
            self._Vrot = scisp.spdiags((cnst.hbar**2.0)/(2.0*self.m)*(l**2+l) *
                                       self.spacegrid.grid_lin[:, 0]**-2, [0],
                                       gn, gn, format='csc')
            self.H = self._K + self._V + self._Vrot

    def eigenstates(self):

        if self.spacegrid.log:
            evals, evecs = sci_lin.eigh(self.H.toarray(), self._R2.toarray())
            evecs *= self.spacegrid.grid_lin[:, 0, None]**0.5
            evecs /= np.linalg.norm(evecs, axis=0)[None, :]
        else:
            evals, evecs = np.linalg.eigh(self.H.toarray())

        return evals, evecs
