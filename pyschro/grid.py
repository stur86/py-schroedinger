# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from pyschro.utils import plane_side


class Grid(object):

    """ A space grid on which to solve the Schroedinger equation of arbitrary
    dimensionality"""

    def __init__(self, bounds, size, log=False):

        bounds = np.array(bounds)
        if len(bounds.shape) != 2 or bounds.shape[1] != 2:
            raise ValueError("Invalid bounds")

        try:
            self.dim = len(size)
            if self.dim != bounds.shape[0]:
                raise ValueError("Invalid size")
        except TypeError:
            if type(size) == int and size > 1:
                self.dim = len(bounds)
                size = (np.ones(self.dim) * size).astype(int)
            else:
                raise ValueError("Invalid size")

        self.bounds = bounds
        self.size = size

        # Now to actually build this
        self._log = log
        if not log:
            self.axes = np.array([np.linspace(b[0], b[1], self.size[i])
                                  for i, b in enumerate(self.bounds)])
            self.dx = np.abs([ax[1] - ax[0] for ax in self.axes])
        else:
            self.axes = np.array([b[0]*np.exp(np.linspace(0, np.log(b[1]/b[0]),
                                                          self.size[i]))
                                  for i, b in enumerate(self.bounds)])
            self.dx = np.abs([ax[1]/ax[0] for ax in self.axes])
        # By this fairly complicated contraption we get to an array where the
        # last coordinate updates faster
        if (self.dim > 1 or int(np.version.version.split('.')[1]) >= 9):
            spacegrid = np.meshgrid(*np.roll(self.axes, -1, axis=0),
                                    indexing='ij')
        else:
            spacegrid = [self.axes, ]
        spacegrid = np.swapaxes(spacegrid, 0, self.dim)
        spacegrid = np.roll(spacegrid, 1, axis=self.dim)
        spacegrid_lin = self.vol2lin(spacegrid)

        self.grid = spacegrid
        self.grid_lin = spacegrid_lin

    @property
    def shape(self):
        return self.grid.shape

    @property
    def linshape(self):
        return self.grid_lin.shape

    @property
    def log(self):
        return self._log

    def vol2lin(self, vol):
        return np.reshape(vol, (int(np.prod(self.size)), -1))

    def lin2vol(self, lin):
        return np.reshape(lin, tuple(list(self.size) + [-1]))

    def ravel_inds(self, inds):
        return np.ravel_multi_index(inds, self.size)

    def unravel_ind(self, ind):
        return np.unravel_index(ind, self.size)

    def validate(self, v, lin=True):
        if lin:
            return np.array(v,
                            copy=False).shape == (int(np.prod(self.size)),)
        else:
            return (np.array(v,
                             copy=False).shape == self.size).all()

    def gradient(self, v, lin=True):
        # Derive the given vector on grid

        # Is it okay?
        if not self.validate(v, lin=lin):
            raise ValueError("Invalid vector passed to gradient")

        # Reshape if linear
        if lin:
            v = np.squeeze(self.lin2vol(v), axis=-1)

        if not self.log:
            grad = np.array(np.gradient(v, *self.dx))
        else:
            grad = np.array(np.gradient(v))
            grad /= np.moveaxis(self.grid*np.log(self.dx), 2, 0)

        if len(grad.shape) == 1:
            grad = grad[None, :]
        grad = np.moveaxis(grad, 0, -1)

        if lin:
            grad = self.vol2lin(grad)

        return grad

    def random_p(self):
        # Random point from inside the bounds of the grid
        p = np.random.random(self.dim)
        p *= self.bounds[:, 1] - self.bounds[:, 0]
        p += self.bounds[:, 0]

        return p

    def plane_side_indices(self, p0, n, side=1, lin=True):

        if lin:
            return np.where(plane_side(self.grid_lin, p0, n) == side)
        else:
            return np.where(plane_side(self.grid, p0, n) == side)
