"""Field-block layout of the global state vector, and restart output.

The global state vector is a concatenation of `FIELD_BLOCK_COUNT` equally sized
field blocks named by `FIELD_NAMES`. Reduction keeps only `KEPT_BLOCK_IDS` and
drops boundary-condition rows.
"""

import os

import h5py
import numpy as np
from numpy.typing import NDArray

from .matrices import HDF5Matrix

FIELD_NAMES: tuple[str, ...] = (
  'U_n', 'U_velx', 'U_vely', 'U_velz', 'U_T', 'U_psi', 'U_by')
FIELD_BLOCK_COUNT = 7
#: Keeps n, velx, velz, T and psi; drops vely (2) and by (6).
KEPT_BLOCK_IDS = (0, 1, 3, 4, 5)


def build_reduction_mapping(jacobian_path: str) -> tuple[int, NDArray[np.bool_]]:
  """Build the global boolean mask used to reduce the Jacobian and mass matrix."""
  jac = HDF5Matrix(jacobian_path, '/jacobian')
  nrg_block = jac.nrg // FIELD_BLOCK_COUNT
  keep_global = np.zeros(jac.nrg, dtype=bool)
  for i in KEPT_BLOCK_IDS:
    keep_global[nrg_block * i:nrg_block * (i + 1)] = True

  if np.any(jac.bcg):
    keep_global &= ~jac.bcg

  return int(jac.nr), keep_global


def write_restart_eigenvectors(
  eigvecs: NDArray[np.complexfloating],
  keep_global: NDArray[np.bool_],
  nr_local: int,
  out_dir: str,
) -> None:
  """Write reduced eigenvectors as restart files, one per eigenvector."""
  # Convert reduced eigenvectors to global, then split into physical fields.
  nred = int(np.count_nonzero(keep_global))
  if eigvecs.shape[0] != nred:
    raise ValueError(
      f'eigvec rows ({eigvecs.shape[0]}) do not match reduced size ({nred})')
  if nr_local % FIELD_BLOCK_COUNT != 0:
    raise ValueError(f'nr_local ({nr_local}) must be divisible by 7')
  if keep_global.shape[0] % FIELD_BLOCK_COUNT != 0:
    raise ValueError(
      f'global vector size ({keep_global.shape[0]}) must be divisible by 7')

  os.makedirs(out_dir, exist_ok=True)

  for i, vec in enumerate(eigvecs.T):
    global_vec = np.zeros((keep_global.shape[0],), dtype=vec.dtype)
    global_vec[keep_global] = vec
    global_vec = np.real(global_vec)
    # Separate the global state vector into its field blocks.
    blocks = np.split(global_vec, FIELD_BLOCK_COUNT)
    with h5py.File(f'{out_dir}/xmhd2d_{i:05d}.rst', 'w') as f:
      f.create_dataset('OFT_idx_Version', data=np.array([1], dtype=np.int32))
      f.create_dataset('t', data=np.array([float(i)], dtype=np.float64))
      f.create_dataset('dt', data=np.array([1.0], dtype=np.float64))
      for j, block in enumerate(blocks):
        f.create_dataset(FIELD_NAMES[j], data=block)
