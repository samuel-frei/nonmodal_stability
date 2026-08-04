"""Build the reduced time-advance operator and its spectral factorisations.

Each stage caches its result to a `.npy` file and silently reuses it when
present. The caches are keyed by filename only, so they are **not** invalidated
when the input matrices, `DEFAULT_TIMESTEP` or `KEPT_BLOCK_IDS` change; delete
them by hand after any such change.
"""

import os
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy.typing import NDArray
from scipy import sparse

from .fields import FIELD_BLOCK_COUNT, write_restart_eigenvectors
from .matrices import HDF5Matrix, assemble_global

REAL_JACOBIAN_CACHE = './real_jacobian.npy'
EIGVAL_CACHE = './full_reduced_eigvals.npy'
EIGVEC_CACHE = './full_reduced_eigvecs.npy'
SCHUR_CACHE = './full_reduced_schur.npy'
DEFAULT_TIMESTEP = 1e-7


def _report_cache_hit(label: str, path: str, shape: tuple[int, ...]) -> None:
  """Log which cache file was reused, and when it was written.

  Caches are keyed by filename alone, so surfacing the absolute path and mtime
  is the cheapest guard against silently computing with a stale operator.
  """
  abs_path = os.path.abspath(path)
  try:
    ts = os.path.getmtime(abs_path)
    mtime = datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec='seconds')
  except OSError:
    mtime = 'unknown'
  print(f'loaded cached {label}: {abs_path} shape={shape} mtime={mtime}', flush=True)


def load_or_compute_jacobian(
  jacobian_path: str,
  massmat_path: str,
  keep_global: NDArray[np.bool_],
) -> NDArray[np.float64]:
  """Load the cached reduced operator, or build and cache it from HDF5 matrices."""
  try:
    real_jac = np.load(REAL_JACOBIAN_CACHE)
    _report_cache_hit('reduced Jacobian', REAL_JACOBIAN_CACHE, real_jac.shape)
    return real_jac
  except (FileNotFoundError, OSError, ValueError):
    pass

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

  np.save(REAL_JACOBIAN_CACHE, real_jac)
  print(f'saved reduced Jacobian cache: {os.path.abspath(REAL_JACOBIAN_CACHE)}', flush=True)
  return real_jac


def load_or_compute_eigvals(
  real_jac: NDArray[np.float64],
  keep_global: NDArray[np.bool_],
  nr_local: int,
  output_dir: str,
) -> NDArray[np.complex128]:
  """Load cached eigenvalues, or compute and cache the spectrum and eigenvectors."""
  try:
    eigvals = np.load(EIGVAL_CACHE)
    _report_cache_hit('eigenvalues', EIGVAL_CACHE, eigvals.shape)
    return eigvals
  except (FileNotFoundError, OSError, ValueError):
    pass

  print('computing full eigenvalue spectrum with numpy.linalg.eigvals', flush=True)
  eigvals = np.asarray(np.linalg.eigvals(real_jac), dtype=np.complex128)
  _, eigvecs = sparse.linalg.eigs(real_jac, k=40, ncv=90, which='LM')
  np.save(EIGVAL_CACHE, eigvals)
  np.save(EIGVEC_CACHE, eigvecs)

  eigvec_dir = os.path.join(output_dir, 'eigvecs_plot')
  write_restart_eigenvectors(eigvecs, keep_global, nr_local, eigvec_dir)

  plt.figure()
  plt.scatter(eigvals.real, eigvals.imag, s=2, c='k')
  plt.savefig('./full_reduced_spectrum.png')
  plt.close()

  return eigvals


def load_or_compute_schur(
  real_jac: NDArray[np.float64],
) -> NDArray[np.complex128]:
  """Load the cached Schur factorisation, or compute and persist it."""
  try:
    schur_t = np.load(SCHUR_CACHE)
    _report_cache_hit('Schur factor', SCHUR_CACHE, schur_t.shape)
    return schur_t
  except (FileNotFoundError, OSError, ValueError):
    pass

  print('computing schur factorization', flush=True)
  schur_raw, _ = scipy.linalg.schur(real_jac, output='complex')
  schur_t = np.asarray(schur_raw, dtype=np.complex128)
  plt.figure()
  plt.scatter(schur_t.diagonal().real, schur_t.diagonal().imag, s=2, c='k')
  plt.savefig('./full_reduced_schur_eigs.png')
  plt.close()
  np.save(SCHUR_CACHE, schur_t)
  return schur_t
