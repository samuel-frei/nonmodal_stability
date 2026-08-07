"""Sparse matrices loaded from HDF5 exports, and global assembly.

* `HDF5Matrix` -- one matrix read from an HDF5 dataset group.
* `assemble_global` -- scatter a local matrix into a dense global one.
"""

from typing import Any

import h5py
import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy import sparse


class HDF5Matrix:
  """A sparse matrix stored as an HDF5 dataset group."""

  def __init__(self, filename: str, mat_name: str) -> None:
    # lc, lg and kr are written one-based by the Fortran writer.
    with h5py.File(filename, 'r') as f:
      self.mat_name: str = mat_name
      self.nr: int = f[f'{mat_name}/nr'][0]
      self.nc: int = f[f'{mat_name}/nc'][0]
      self.nrg: int = f[f'{mat_name}/nrg'][0]
      self.ncg: int = f[f'{mat_name}/ncg'][0]
      self.lc: NDArray[np.int64] = np.array(f[f'{mat_name}/lc']) - 1
      self.lg: NDArray[np.int64] = np.array(f[f'{mat_name}/lg']) - 1
      self.kr: NDArray[np.int64] = np.array(f[f'{mat_name}/kr']) - 1
      self.M: NDArray[np.float64] = np.array(f[f'{mat_name}/M'])
      try:
        self.bc_flags: NDArray[np.bool_] = np.array(f[f'{mat_name}/bc_flags']) > 0
        self.bcg: NDArray[np.bool_] = np.zeros((self.nrg,), dtype=bool)
        self.bcg[self.lg] = self.bc_flags
      except KeyError:
        # Matrices without boundary-condition flags carry no constrained rows.
        self.bc_flags = np.zeros(self.nr, dtype=bool)
        self.bcg = np.zeros((self.nrg,), dtype=bool)
    self.csr_rep: Any = sparse.csr_array((self.M, self.lc, self.kr))


@njit
def assemble_global(
  inmat: NDArray[np.float64],
  nrg: int,
  ncg: int,
  lg: NDArray[np.int64],
) -> NDArray[np.float64]:
  """Scatter a local matrix into a dense global matrix using the `lg` mapping."""
  # Both loop bounds come from inmat.shape[0], so a rectangular input would be
  # mis-assembled rather than rejected.
  if inmat.shape[0] != inmat.shape[1]:
    raise ValueError('assemble_global requires a square local matrix')
  outmat = np.zeros((nrg, ncg))
  for i in range(inmat.shape[0]):
    for j in range(inmat.shape[0]):
      ik = lg[i]
      jl = lg[j]
      outmat[ik, jl] += inmat[i, j]
  return outmat
