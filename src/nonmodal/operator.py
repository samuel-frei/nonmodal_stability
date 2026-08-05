"""Build the reduced time-advance operator and its spectral factorisations.

Each stage caches its result to a `.npy` file and silently reuses it when
present. The caches are keyed by filename only, so they are **not** invalidated
when the input matrices, `DEFAULT_TIMESTEP` or `KEPT_BLOCK_IDS` change; delete
them by hand after any such change.
"""

import os
from datetime import UTC, datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy.typing import NDArray
from scipy import sparse

from .fields import FIELD_BLOCK_COUNT, write_restart_eigenvectors
from .matrices import HDF5Matrix, assemble_global

REAL_JACOBIAN_CACHE = 'real_jacobian.npy'
EIGVAL_CACHE = 'full_reduced_eigvals.npy'
EIGVEC_CACHE = 'full_reduced_eigvecs.npy'
SCHUR_CACHE = 'full_reduced_schur.npy'
SPECTRUM_PLOT = 'full_reduced_spectrum.png'
SCHUR_PLOT = 'full_reduced_schur_eigs.png'
DEFAULT_TIMESTEP = 1e-7
DEFAULT_CACHE_DIR = '.'


def _load_cached(cache_dir: str, filename: str, label: str) -> NDArray[Any] | None:
  """Return a cached array, or None if it is absent or unreadable.

  Caches are keyed by filename alone, so a hit logs the absolute path and mtime:
  that is the cheapest guard against silently computing with a stale operator
  after the inputs, DEFAULT_TIMESTEP or KEPT_BLOCK_IDS have changed.
  """
  path = os.path.abspath(os.path.join(cache_dir, filename))
  try:
    array = np.load(path)
  except (FileNotFoundError, OSError, ValueError):
    return None
  try:
    mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=UTC).isoformat(
      timespec='seconds')
  except OSError:
    mtime = 'unknown'
  print(f'loaded cached {label}: {path} shape={array.shape} mtime={mtime}', flush=True)
  return array


def _save_cached(cache_dir: str, filename: str, array: NDArray[Any]) -> None:
  """Persist an array into the cache directory, creating it if needed."""
  os.makedirs(cache_dir, exist_ok=True)
  np.save(os.path.join(cache_dir, filename), array)


def load_or_compute_jacobian(
  jacobian_path: str,
  massmat_path: str,
  keep_global: NDArray[np.bool_],
  cache_dir: str = DEFAULT_CACHE_DIR,
) -> NDArray[np.float64]:
  """Load the cached reduced operator, or build and cache it from HDF5 matrices."""
  cached = _load_cached(cache_dir, REAL_JACOBIAN_CACHE, 'reduced Jacobian')
  if cached is not None:
    return cached

  mmat = HDF5Matrix(massmat_path, '/massmat')
  jac = HDF5Matrix(jacobian_path, '/jacobian')

  mmat_big = sparse.block_diag([mmat.csr_rep] * FIELD_BLOCK_COUNT, format='csr')
  del mmat

  jac_dense = assemble_global(jac.csr_rep.toarray(), nrg=jac.nrg, ncg=jac.ncg, lg=jac.lg)
  jac_gl = sparse.csr_array(jac_dense)
  jac_gl.eliminate_zeros()
  del jac_dense

  mmat_dense = assemble_global(mmat_big.toarray(), nrg=jac.nrg, ncg=jac.ncg, lg=jac.lg)
  mmat_gl = sparse.csr_array(mmat_dense)
  mmat_gl.eliminate_zeros()
  del mmat_dense

  print('Shape of jac_gl is:', jac_gl.shape, flush=True)
  print('shape of mmat_gl is:', mmat_gl.shape, flush=True)

  reduced_jac = jac_gl[keep_global][:, keep_global].tocsr()
  reduced_mmat = mmat_gl[keep_global][:, keep_global].tocsr()

  print('Shape of reduced_Jac is:', reduced_jac.shape, flush=True)
  print('shape of reduced_Mmat is:', reduced_mmat.shape, flush=True)

  solved = sparse.linalg.spsolve(reduced_jac, reduced_mmat.toarray())
  solved = np.asarray(solved)
  identity = np.eye(reduced_jac.shape[0], dtype=solved.dtype)
  real_jac = (solved - identity) / DEFAULT_TIMESTEP

  _save_cached(cache_dir, REAL_JACOBIAN_CACHE, real_jac)
  return real_jac


def load_or_compute_eigvals(
  real_jac: NDArray[np.float64],
  keep_global: NDArray[np.bool_],
  nr_local: int,
  output_dir: str,
  cache_dir: str = DEFAULT_CACHE_DIR,
) -> NDArray[np.complex128]:
  """Load cached eigenvalues, or compute and cache the spectrum and eigenvectors."""
  cached = _load_cached(cache_dir, EIGVAL_CACHE, 'eigenvalues')
  if cached is not None:
    return np.asarray(cached, dtype=np.complex128)

  print('computing full eigenvalue spectrum with numpy.linalg.eigvals', flush=True)
  eigvals = np.asarray(np.linalg.eigvals(real_jac), dtype=np.complex128)
  _, eigvecs = sparse.linalg.eigs(real_jac, k=40, ncv=90, which='LM')
  _save_cached(cache_dir, EIGVAL_CACHE, eigvals)
  _save_cached(cache_dir, EIGVEC_CACHE, eigvecs)

  eigvec_dir = os.path.join(output_dir, 'eigvecs_plot')
  write_restart_eigenvectors(eigvecs, keep_global, nr_local, eigvec_dir)

  plt.figure()
  plt.scatter(eigvals.real, eigvals.imag, s=2, c='k')
  plt.savefig(os.path.join(cache_dir, SPECTRUM_PLOT))
  plt.close()

  return eigvals


def load_or_compute_schur(
  real_jac: NDArray[np.float64],
  cache_dir: str = DEFAULT_CACHE_DIR,
) -> NDArray[np.complex128]:
  """Load the cached Schur factorisation, or compute and persist it."""
  cached = _load_cached(cache_dir, SCHUR_CACHE, 'Schur factor')
  if cached is not None:
    return np.asarray(cached, dtype=np.complex128)

  print('computing schur factorization', flush=True)
  schur_raw, _ = scipy.linalg.schur(real_jac, output='complex')
  schur_t = np.asarray(schur_raw, dtype=np.complex128)
  os.makedirs(cache_dir, exist_ok=True)
  plt.figure()
  plt.scatter(schur_t.diagonal().real, schur_t.diagonal().imag, s=2, c='k')
  plt.savefig(os.path.join(cache_dir, SCHUR_PLOT))
  plt.close()
  _save_cached(cache_dir, SCHUR_CACHE, schur_t)
  return schur_t
