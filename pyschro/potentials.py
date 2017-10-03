# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.constants as cnst


class Potential(object):

    @staticmethod
    def default_pars():
        return {}

    def __init__(self, eVAng=True, **in_pars):

        pars = self.default_pars()
        for k in in_pars:
            try:
                pars[k] = in_pars[k]
            except KeyError:
                raise ValueError("Invalid parameter passed to potential")

        if eVAng:
            pars = self.toSI(**pars)

        self._V = self.makeV(**pars)
        self._F = self.makeF(**pars)

    @property
    def V(self):
        return self._V

    @property
    def F(self):
        return self._F

    def __add__(self, other):
        # Add two potentials together

        ans = Potential()

        try:
            ans._V = lambda x: self._V(x) + other._V(x)
            ans._F = lambda x: self._F(x) + other._F(x)
        except AttributeError:
            raise TypeError('Unsupported operand + for this type')

        return ans

    def __sub__(self, other):
        # Subtract two potentials

        ans = Potential()

        try:
            ans._V = lambda x: self._V(x) - other._V(x)
            ans._F = lambda x: self._F(x) - other._F(x)
        except AttributeError:
            raise TypeError('Unsupported operand + for this type')

        return ans

    # Default is free potential, so...
    def toSI(self):
        return {}

    def makeV(self):
        return lambda x: 0.0*np.linalg.norm(x, axis=-1)

    def makeF(self):
        return lambda x: 0.0*x


class ConstantPotential(Potential):

    @staticmethod
    def default_pars():
        return {'V0': 0.0}

    def toSI(self, V0):
        return {'V0': V0*cnst.electron_volt}

    def makeV(self, V0):
        return lambda x: np.ones(x.shape[:-1])*V0

    def makeF(self, V0):
        return lambda x: np.zeros(x.shape[:-1])


class BarrierPotential(Potential):

    @staticmethod
    def default_pars():
        return {'p0_b': 0.0, 'n_b': [1], 'w_b': 0.1, 'V_b': 1.0}

    def toSI(self, p0_b, n_b, w_b, V_b):
        return {'p0_b': np.array(p0_b)*1e-10,
                'n_b': np.array(n_b)/np.linalg.norm(n_b),
                'w_b': w_b*1e-10,
                'V_b': V_b*cnst.electron_volt}

    def makeV(self, p0_b, n_b, w_b, V_b):
        return lambda x: V_b*np.exp(-np.tensordot((x-p0_b)/w_b, n_b,
                                                  axes=(-1, -1))**2)

    def makeF(self, p0_b, n_b, w_b, V_b):
        return lambda x: 2*V_b*np.tensordot((x-p0_b)/w_b**2, n_b,
                                            axes=(-1, -1))[:, None] *\
            n_b[None, :] *\
            np.exp(-np.tensordot((x-p0_b)/w_b, n_b,
                                 axes=(-1, -1))**2)[:, None]


class HarmonicPotential(Potential):

    @staticmethod
    def default_pars():
        return {'k': 1.0, 'x0': 0.0}

    def toSI(self, k, x0):
        return {'k': k*cnst.electron_volt*1e20,
                'x0': np.array(x0)*1e-10}

    def makeV(self, k, x0):
        return lambda x: k*np.linalg.norm(x-x0, axis=-1)**2

    def makeF(self, k, x0):
        return lambda x: -2*k*(x-x0)


class HarmonicAxisPotential(Potential):

    @staticmethod
    def default_pars():
        return {'k': [1.0], 'x0': 0.0}

    def toSI(self, k, x0):
        return {'k': np.array(k)*cnst.electron_volt*1e20,
                'x0': np.array(x0)*1e-10}

    def makeV(self, k, x0):
        kval = np.linalg.norm(k)
        kdir = k/kval
        return lambda x: kval*np.tensordot(x-x0, kdir, axes=(-1, 0))**2.0

    def makeF(self, k, x0):
        kval = np.linalg.norm(k)
        kdir = k/kval
        return lambda x: -2*kval*np.tensordot(x-x0, kdir, axes=(-1, 0))*kdir


class DoubleWellPotential(Potential):

    @staticmethod
    def default_pars():
        return {'Dx': 0.1,
                'DV': 1.0,
                'asymmV': 0.0,
                'x0': 0.0}

    def toSI(self, Dx, DV, asymmV, x0):
        return {'Dx': Dx*1e-10,
                'DV': DV*cnst.electron_volt,
                'asymmV': asymmV*cnst.electron_volt,
                'x0': np.array(x0)*1e-10
                }

    def makeV(self, Dx, DV, asymmV, x0):
        _B = 2.0*DV/Dx**2.0
        _A = _B/(2.0*Dx**2.0)

        return lambda x: _A*np.linalg.norm(x-x0, axis=-1)**4 -\
            _B*np.linalg.norm(x-x0, axis=-1)**2 +\
            np.sum(asymmV*(x-x0)/Dx, axis=-1)+DV

    def makeF(self, Dx, DV, asymmV, x0):
        _B = 2.0*DV/Dx**2.0
        _A = _B/(2.0*Dx**2.0)

        return lambda x: -4*_A*(x-x0)*np.expand_dims(np.sum((x-x0)*(x-x0),
                                                            axis=-1),
                                                     axis=-1) +\
            2*_B*(x-x0)-asymmV*np.ones(x.shape)/Dx


class StepRadialPotential(Potential):

    @staticmethod
    def default_pars():
        return {'w': 0.1,
                'x0': 0.0,
                'r0': 1.0,
                'DV': 1.0}

    def toSI(self, w, x0, r0, DV):
        return {'w': w*1e-10,
                'x0': np.array(x0)*1e-10,
                'r0': r0*1e-10,
                'DV': DV*cnst.electron_volt
                }

    def makeV(self, w, x0, r0, DV):

        return lambda x: DV/(1.0+np.exp(-(np.linalg.norm(x-x0, axis=-1)-r0)/w))

    def makeF(self, w, x0, r0, DV):

        def funcF(x):
            nx = np.linalg.norm(x-x0, axis=-1)
            expx = np.exp(-(nx-r0)/w)
            return (DV/(1.0+expx)**2.0*expx/w)[:, None]*(x-x0)/nx[:, None]

        return funcF


class TrapPotential(Potential):

    @staticmethod
    def default_pars():
        return {'DxMax': 0.1,
                'DxMin': 0.2,
                'DV': 1.0,
                'x0': 0.0}

    def toSI(self, DxMax, DxMin, DV, x0):
        return {'DxMax': DxMax*1e-10,
                'DxMin': DxMin*1e-10,
                'DV': DV*cnst.electron_volt,
                'x0': np.array(x0)*1e-10
                }

    def makeV(self, DxMax, DxMin, DV, x0):
        # Solve the equation
        M = np.array([[DxMin**6, -DxMin**4, DxMin**2],
                      [6*DxMin**5, -4*DxMin**3, 2*DxMin],
                      [6*DxMax**5, -4*DxMax**3, 2*DxMax]])
        v = np.array([DV, 0, 0])
        ABC = np.linalg.solve(M, v)
        if ABC[0] < 0:
            ABC *= -1
        _A, _B, _C = ABC

        return lambda x: _A*np.linalg.norm(x-x0, axis=-1)**6 -\
            _B*np.linalg.norm(x-x0, axis=-1)**4 +\
            _C*np.linalg.norm(x-x0, axis=-1)**2-DV

    def makeF(self, DxMax, DxMin, DV, x0):
        # Solve the equation
        M = np.array([[DxMin**6, -DxMin**4, DxMin**2],
                      [6*DxMin**5, -4*DxMin**3, 2*DxMin],
                      [6*DxMax**5, -4*DxMax**3, 2*DxMax]])
        v = np.array([DV, 0, 0])
        ABC = np.linalg.solve(M, v)
        if ABC[0] < 0:
            ABC *= -1
        _A, _B, _C = ABC

        return lambda x: -6*_A*(x-x0) *\
            np.expand_dims(np.linalg.norm((x-x0),
                                          axis=-1)**4,
                           axis=-1) +\
            4*_B*(x-x0)*np.expand_dims(np.sum((x-x0)*(x-x0),
                                              axis=-1),
                                       axis=-1) -\
            2*_C*(x-x0)


class TwoSitePotential(Potential):

    @staticmethod
    def default_pars():
        return {'DV': 1.0,
                'x0': 1.0,
                'k': 1.0}

    def toSI(self, DV, x0, k):
        return {'DV': DV*cnst.electron_volt,
                'x0': x0*1e-10,
                'k': k*cnst.electron_volt*1e20}

    def makeV(self, DV, x0, k):

        def tsV(x):
            r0 = np.zeros(x[0].shape)
            r0[0] = x0
            dx = np.where(x[:, 0, None] > 0, x-r0, x+r0)
            return k/2.0*np.sum(dx**2, axis=1)

        return tsV

    def makeF(self, DV, x0, k):

        def tsF(x):
            r0 = np.zeros(x[0].shape)
            r0[0] = x0
            dx = np.where(x[:, 0, None] > 0, x-r0, x+r0)
            return -k*dx

        return tsF


class LennardJonesPotential(Potential):

    @staticmethod
    def default_pars():
        return {'DV': 1.0,
                'r0': 1.0,
                'x0': 0.0,
                'n': None,
                'A': 0}

    def toSI(self, DV, r0, x0, n, A):
        return {'DV': DV*cnst.electron_volt,
                'r0': r0*1e-10,
                'x0': x0*1e-10,
                'n': None if n is None else np.array(n)/np.linalg.norm(n),
                'A': A*cnst.electron_volt
                }

    def makeV(self, DV, r0, x0, n, A):

        def ljV(x):
            r = (x-x0)/r0
            R = np.linalg.norm(r, axis=-1)
            R6 = R**-6
            V = DV*(R6**2-2*R6)
            if n is not None:
                rcos = np.dot(r, n)/np.linalg.norm(r, axis=-1)
                return V - A*(rcos-1.0)
            else:
                return V

        return ljV

    def makeF(self, DV, r0, x0, n, A):

        def ljF(x):
            r = (x-x0)/r0
            R = np.linalg.norm(r, axis=-1)
            Fr = DV*12*r*((R**-14-R**-8)[:, None])/r0
            if n is not None:
                rcos = np.dot(r, n)/R
                return Fr+A*(n[None, :]/R[:, None]-rcos[:, None] *
                             r/(R**2)[:, None])/r0
            else:
                return Fr

        return ljF
