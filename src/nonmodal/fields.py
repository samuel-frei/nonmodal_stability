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


def _check_restart_shapes(
  vectors: NDArray[np.complexfloating],
  keep_global: NDArray[np.bool_],
  nr_local: int,
) -> None:
  """Reject vectors that cannot be scattered into the global field layout."""
  nred = int(np.count_nonzero(keep_global))
  if vectors.shape[0] != nred:
    raise ValueError(
      f'eigvec rows ({vectors.shape[0]}) do not match reduced size ({nred})')
  if nr_local % FIELD_BLOCK_COUNT != 0:
    raise ValueError(f'nr_local ({nr_local}) must be divisible by 7')
  if keep_global.shape[0] % FIELD_BLOCK_COUNT != 0:
    raise ValueError(
      f'global vector size ({keep_global.shape[0]}) must be divisible by 7')


def _write_restart(
  out_dir: str,
  index: int,
  values: NDArray[np.floating],
  keep_global: NDArray[np.bool_],
) -> str:
  """Scatter one reduced real vector into the global layout and write it out.

  `t` is the file's own index rather than a physical time, which is what lets a
  directory of these be read back as a sequence.
  """
  global_vec = np.zeros((keep_global.shape[0],), dtype=np.float64)
  global_vec[keep_global] = values
  # Separate the global state vector into its field blocks.
  blocks = np.split(global_vec, FIELD_BLOCK_COUNT)
  path = f'{out_dir}/xmhd2d_{index:05d}.rst'
  with h5py.File(path, 'w') as f:
    f.create_dataset('OFT_idx_Version', data=np.array([1], dtype=np.int32))
    f.create_dataset('t', data=np.array([float(index)], dtype=np.float64))
    f.create_dataset('dt', data=np.array([1.0], dtype=np.float64))
    for j, block in enumerate(blocks):
      f.create_dataset(FIELD_NAMES[j], data=block)
  return path


def aligned_phase(vec: NDArray[np.complexfloating]) -> float:
  """The phase `theta` maximising `||Re(vec * exp(-i*theta))||`.

  A complex mode has an arbitrary overall phase -- ARPACK's is whatever the
  iteration happened to land on -- so taking the real part directly can discard
  most of the amplitude, in the worst case all of it. Writing
  `Re(v) cos(theta) + Im(v) sin(theta)` and maximising over `theta` gives
  `tan(2*theta) = 2 a.b / (a.a - b.b)` for `a = Re v`, `b = Im v`.
  """
  a = np.asarray(vec.real, dtype=np.float64)
  b = np.asarray(vec.imag, dtype=np.float64)
  return 0.5 * float(np.arctan2(2.0 * float(a @ b), float(a @ a) - float(b @ b)))


def write_restart_eigenvectors(
  eigvecs: NDArray[np.complexfloating],
  keep_global: NDArray[np.bool_],
  nr_local: int,
  out_dir: str,
) -> None:
  """Write reduced eigenvectors as restart files, one per eigenvector.

  Takes the real part as-is, at whatever phase the eigensolver produced. Kept
  exactly as it was so existing runs stay reproducible; `write_restart_modes`
  is the phase-aware path.
  """
  _check_restart_shapes(eigvecs, keep_global, nr_local)
  os.makedirs(out_dir, exist_ok=True)
  for i, vec in enumerate(eigvecs.T):
    _write_restart(out_dir, i, np.real(vec), keep_global)


def write_restart_modes(
  vectors: NDArray[np.complexfloating],
  keep_global: NDArray[np.bool_],
  nr_local: int,
  out_dir: str,
  phases: int = 1,
) -> list[str]:
  """Write complex modes as restart files, optionally sweeping their phase.

  `phases=1` writes one file per mode at the phase carrying the most amplitude.
  Larger values sweep `[0, 2*pi)` from that phase, and since `_write_restart`
  stores each file's index as `t`, the directory reads back as a time series --
  a travelling mode animates with no change to the format.

  Returns the paths written, in order.
  """
  _check_restart_shapes(vectors, keep_global, nr_local)
  if phases < 1:
    raise ValueError('phases must be >= 1')

  os.makedirs(out_dir, exist_ok=True)
  offsets = np.linspace(0.0, 2.0 * np.pi, phases, endpoint=False)

  paths: list[str] = []
  for vec in vectors.T:
    theta0 = aligned_phase(vec)
    for offset in offsets:
      rotated = np.real(vec * np.exp(-1j * (theta0 + offset)))
      paths.append(_write_restart(out_dir, len(paths), rotated, keep_global))
  return paths
