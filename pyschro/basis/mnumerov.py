# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.constants as cnst
import scipy.sparse as scisp
import scipy.sparse.linalg as scisp_lin

from pyschro.basis.basis import BasisSet
from pyschro.utils import multikron


class MNumerovBasis(BasisSet):

    def __init__(self, grid, V, m, periodic=False, bfunc=None):

        super(MNumerovBasis, self).__init__(grid, V, m)

        # Now do our own thing!

        # This is a bit messy so we go by steps. First, we generate the
        # differentiation operators for each axis
        # Sparse matrices make this SO MUCH faster

        D2x_base = []
        for gn in grid.size:
            D2x_base.append(scisp.spdiags([[1.0]*gn, [-2.0]*gn, [1.0]*gn],
                                          (-1, 0, 1), gn, gn, format='csc'))
            if (periodic):
                D2x_base[-1][0, -1] += D2x_base[-1][0, 1]
                D2x_base[-1][-1, 0] += D2x_base[-1][-1, -2]

        D2x_list = []
        for d, base in enumerate(D2x_base):
            D2x = 1.0/grid.dx[d]**2.0*base
            D2x = multikron([D2x if i == d else np.identity(grid.size[i])
                             for i in range(grid.dim)])
            D2x_list.append(D2x)

        # Now to build the pair products
        D2xy_list = []
        for d1 in range(grid.dim):
            for d2 in range(d1+1, grid.dim):
                D2xy_list.append((grid.dx[d1]**2.0+grid.dx[d2]**2.0) *
                                 (D2x_list[d1]*D2x_list[d2]))

        # Ok, full finite difference operator
        # This corresponds to Lapl(f) + 1/12*Lapl(Lapl(f)), and thus the
        # Numerov algorithm comes from it

        D2 = sum(D2x_list) + 1.0/12.0*sum(D2xy_list)

        # Now we build the full kinetic operator
        A = D2
        B = scisp.identity(len(self.V), format='csc')
        B += sum([grid.dx[d]**2.0/12.0*D2x_list[d] for d in range(grid.dim)])
        if bfunc is not None:
            if hasattr(bfunc, '__call__'):
                bfunc = bfunc(grid.grid_lin)
            B = B*scisp.spdiags([bfunc], (0,), len(bfunc), len(bfunc),
                                format='csc')
        K = -(cnst.hbar**2.0)/(2.0*m)*(scisp_lin.inv(B)*A)

        # And here's the Hamiltonian
        self.H = K + scisp.spdiags((self.V,), [0], len(self.V), len(self.V),
                                   format='csc')
