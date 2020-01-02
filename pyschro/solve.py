# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple

try:
    import qutip as qu
except ImportError:
    qu = None
import numpy as np
import scipy.constants as cnst
import scipy.sparse as scisp
import scipy.sparse.linalg as scisp_lin

from pyschro.grid import Grid
from pyschro.basis import BasisSet, MNumerovBasis

QEvol = namedtuple('QEvol', ['params', 't', 'E', 'density'])


class QSolution(object):

    def __init__(self, bounds, size, V, m,
                 basis=MNumerovBasis,
                 basis_pars={},
                 n_states=None):

        self.grid = Grid(bounds, size)
        # Check that the basis type matches
        if not issubclass(basis, BasisSet):
            raise ValueError("Invalid basis set!")
        self.basis = basis(self.grid, V, m, **basis_pars)

        if n_states is None or n_states >= len(self.grid.grid_lin):
            if hasattr(self.basis.H, 'toarray'):
                H = self.basis.H.toarray()
            else:
                H = self.basis.H
            try:
                assert(np.isclose(H-H.conjugate().T, 0).all())
            except AssertionError:
                raise RuntimeError("Hamiltonian is not Hermitian, "
                                   "something has gone terribly wrong!")
            self.evals, self.evecs = np.linalg.eigh(H)
        else:
            H = self.basis.H
            try:
                HHcheck = (H-H.conjugate().T)
                if hasattr(HHcheck, 'todense'):
                    HHcheck = HHcheck.todense()
                assert(np.isclose(HHcheck, 0).all())
            except AssertionError:
                raise RuntimeError("Hamiltonian is not Hermitian, "
                                   "something has gone terribly wrong!")
            (self.evals,
             self.evecs) = scisp_lin.eigsh(H,
                                           k=n_states,
                                           which='LM',
                                           sigma=np.amin(self.basis.V))

        self.N = len(self.evals)

        # Evecs in SPAAAAACE!
        self.evecs_grid = self.basis.basis2grid(self.evecs)

    def evec_grid(self, i):
        # Return evector i expressed in real space
        return self.evecs_grid[:, i]

    def P_grid(self, i):
        # Return probability density in real space for evector i
        ev = self.evec_grid(i)
        return np.abs(ev)**2

    def partition_function(self, T):

        if not hasattr(self, 'T') or self.T != T:
            self.T = T
            if T != 0:
                self.Z = np.exp(-(self.evals-self.evals[0])/(cnst.k*T))
                self.Z /= np.sum(self.Z)
            else:
                self.Z = self.evals*0
                self.Z[0] = 1.0

        return self.Z

    def density(self, T):

        if not hasattr(self, '_density') or self.T != T:

            self.partition_function(T)

            # So build the thermal wavefunction, in space
            pvecs = np.abs(self.evecs_grid)**2
            self._density = np.dot(pvecs, self.Z)
            self._density /= np.sum(self._density)
            self.E = np.sum(self.evals*self.Z)

        return self._density

    def psithermal(self, T, phi_vec=None):

        if not hasattr(self, 'T') or self.T != T:
            self.partition_function(T)

        # Complete the phi_vec
        if phi_vec is None:
            phi_vec = np.zeros(len(self.evals))
        else:
            phi_vec = np.array(phi_vec)
            if phi_vec.shape[0] < len(self.evals):
                phi_vec = np.pad(phi_vec,
                                 (0, len(self.evals)-len(phi_vec)),
                                 mode='constant')
            else:
                phi_vec = phi_vec[:len(self.evals)]

        # So build the thermal wavefunction, in space
        psiT = np.dot(self.evecs_grid,
                      np.sqrt(self.Z)*np.exp(1.0j*phi_vec))
        psiT /= np.linalg.norm(psiT)
        self.E = np.sum(self.evals*self.Z)

        return psiT

    def density_of(self, psi):

        # Normalize it
        psi /= np.linalg.norm(psi)
        psigrid = np.dot(self.evecs_grid, psi)
        return np.abs(psigrid)**2

    def evolve(self, psi0=None, psi0grid=None,
               psi0basis=None,
               dt=1e-18, t_steps=1000,
               T=0, gamma=0.0):

        if qu is None:
            raise RuntimeError('A qutip installation is necessary for '
                               'integrating the master equation')

        # First, define the starting wavefunction
        Umat_left = qu.Qobj(self.evecs)
        Umat_right = Umat_left.dag()

        # Starting wavefunction
        if psi0 is not None:
            psi0 = np.array(psi0)
            if len(psi0.shape) != 1:
                raise ValueError("Invalid psi0 passed to evolve")
            if psi0.shape[0] < self.N:
                psi0 = np.pad(psi0,
                              ((0, self.N-psi0.shape[0])),
                              mode=str('constant'))
            else:
                psi0 = psi0[:self.N]
        elif psi0grid is not None:
            psi0grid = np.array(psi0grid)
            if len(psi0grid.shape) != 1:
                raise ValueError("Invalid psi0grid passed to evolve")
            psi0 = np.dot(psi0grid, self.evecs_grid)
        elif psi0basis is not None:
            psi0basis = np.array(psi0basis)
            if len(psi0basis.shape) != 1:
                raise ValueError("Invalid psi0basis passed to evolve")
            psi0 = np.dot(self.basis.basis2grid(psi0basis), self.evecs_grid)
        else:
            raise ValueError("At least one psi0 must be provided to evolve")

        # Normalize it
        psi0 = qu.Qobj(psi0).unit()
        rho0 = qu.ket2dm(psi0)

        # Is this an evolution with dissipation or not?
        c_ops = []      # Collapse operators list starts empty
        if (gamma > 0.0):
            self.partition_function(T)
            # First convert it from 1/s
            gamma *= cnst.hbar/cnst.electron_volt
            # Now create the list of collapse operators
            for i in range(1, self.N):
                Vi = np.zeros((self.N, self.N))
                Vi[0, i] = 1.0
                Vi = qu.Qobj(Vi)
                # Now on to building the actual collapse operator
                c = np.sqrt(gamma)*(np.sqrt(self.Z[0])*Vi +
                                    np.sqrt(self.Z[i])*Vi.dag())
                c_ops.append(c)

        # Time evolution!

        H = qu.Qobj(np.diag(self.evals)/cnst.electron_volt)
        tlist = np.linspace(dt, t_steps*dt,
                            t_steps)*cnst.electron_volt/cnst.hbar

        opts = qu.Odeoptions()
        opts.nsteps = 10000
        results = qu.mesolve(H, rho0, tlist,
                             c_ops=c_ops, e_ops=[], options=opts)

        rhoX = Umat_left*rho0*Umat_right

        evol_t = [0]
        evol_E = [np.real(qu.expect(H, rho0))*cnst.electron_volt]
        evol_d = [np.abs(np.diag(rhoX.full()))]

        for i, t in enumerate(tlist):

            rho = results.states[i]
            E = np.real(qu.expect(H, rho))  # Energy
            # Transform this back into real space
            rhoX = Umat_left*rho*Umat_right

            evol_t.append(t*cnst.hbar/cnst.electron_volt)    # Back to seconds
            evol_E.append(E*cnst.electron_volt)              # Back to J
            evol_d.append(np.abs(np.diag(rhoX.full())))

        evol_t = np.array(evol_t)
        evol_E = np.array(evol_E)
        evol_d = np.array(evol_d)

        self.evol = QEvol({'T': T, 'gamma': gamma}, evol_t, evol_E, evol_d)

        return self.evol
