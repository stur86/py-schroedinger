# py-schroedinger
A Python package for solving the one-particle Schroedinger equation on any dimension using Numpy and Scipy 

This was done for a research project but should be of general enough usefulness, though still not well documented. It provides at the moment two basis sets, plane waves and direct space with a Numerov inspired integration method. References to the Matrix Numerov method used can be found [here](https://www.physics.wisc.edu/~tgwalker/106.Numerov.pdf) for the 1D case and a full description of the method in N dimensions is available [here](https://arxiv.org/pdf/1709.09880.pdf).

For a basic usage example, this script will produce the eigen energies for a proton in a harmonic potential of constant k = 2 eV/Angstrom.

```python
import numpy as np
import scipy.constants as cnst
from pyschro.solve import QSolution

k = 2 # eV/Ang^2

def V(r):
  return (2*1e20*cnst.electron_volt)*np.linalg.sum(r**2, axis=-1)

sol = QSolution([[-3e-10, 3e-10]],      # Bounds: array of N pairs, one for each dimension, in meters
                300,                    # Grid points
                V,                      # Potential function: takes a (grid, N) array in meters, returns a (grid,) one in J
                cnst.m_p)               # Mass in kg; here a proton
                
# Default basis set is direct space

print 'Eigenvalues: '
for i, e in enumerate(sol.evals):
  print 'n = {0}, E = {1} eV'.format(i+1, e/cnst.electron_volt)

```
