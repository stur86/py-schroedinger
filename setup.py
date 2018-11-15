#!/usr/bin/env python
"""
Py-Schroedinger - Schroedinger Equation numerical solution library

"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from setuptools import setup

setup(name='PySchroedinger',
      version='0.9',
      packages=['pyschro'],
      install_requires=[
          'numpy>=1.11',
          'scipy',
          'qutip'])
