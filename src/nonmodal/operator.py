"""Build the reduced time-advance operator and its spectral factorisations.

Each stage caches to `.npy` and reuses it blindly. Caches are keyed by filename
only, so they are NOT invalidated when inputs, the timestep or `KEPT_BLOCK_IDS`
change -- delete them by hand.

* `load_or_compute_jacobian` -- the reduced effective operator `A`.
* `load_or_compute_schur` -- the triangular factor `T` of `A = Z T Z*`.
* `load_or_compute_schur_vectors` -- both `T` and the unitary `Z`.
* `spectrum_from_schur` -- the eigenvalues, which are the diagonal of `T`.
* `rightmost_indices` / `eigenvectors_from_schur` -- eigenvectors by
  back-substitution in `T`, mapped back through `Z`.
* `write_eigenmode_restarts` -- the rightmost eigenvectors, out as `.rst`.
"""

import os
from datetime import UTC, datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy.typing import NDArray
from scipy import sparse

from .adapters import find_input_deck, read_adiabatic_index
from .fields import FIELD_BLOCK_COUNT, TEMPERATURE_BLOCK_ID, write_restart_eigenvectors
from .matrices import HDF5Matrix, assemble_global

REAL_JACOBIAN_CACHE = 'real_jacobian.npy'
SCHUR_CACHE = 'full_reduced_schur.npy'
#: The unitary Z of A = Z T Z*, needed to map a pseudomode back to the basis.
SCHURVEC_CACHE = 'full_reduced_schurvecs.npy'
SCHUR_PLOT = 'full_reduced_schur_eigs.png'
DEFAULT_TIMESTEP = 1e-7
DEFAULT_CACHE_DIR = '.'


def _load_cached(cache_dir: str, filename: str, label: str) -> NDArray[Any] | None:
  """Return a cached array, or None if it is absent or unreadable."""
  # A hit logs path and mtime; the cache key is the filename alone.
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


def mass_blocks(scalar_mass: Any, gamma: float) -> list[Any]:
  """The seven field blocks of the global mass matrix, given the scalar one.

  Only temperature differs: its evolution carries a factor `1/(gamma - 1)`.
  """
  # The export is the scalar Lagrange mass matrix, so no per-field factor can
  # be present in it; OFT applies this one when it assembles its own operator.
  blocks = [scalar_mass] * FIELD_BLOCK_COUNT
  blocks[TEMPERATURE_BLOCK_ID] = scalar_mass / (gamma - 1.0)
  return blocks


def load_or_compute_jacobian(
  jacobian_path: str,
  massmat_path: str,
  keep_global: NDArray[np.bool_],
  cache_dir: str = DEFAULT_CACHE_DIR,
  timestep: float = DEFAULT_TIMESTEP,
) -> NDArray[np.float64]:
  """Load the cached reduced operator, or build and cache it from HDF5 matrices."""
  cached = _load_cached(cache_dir, REAL_JACOBIAN_CACHE, 'reduced Jacobian')
  if cached is not None:
    return cached

  # gamma defines the operator, so it comes from the deck that ran the export.
  gamma = read_adiabatic_index(find_input_deck(jacobian_path))
  print(f'temperature mass block scaled by 1/(gamma-1), gamma={gamma:g}', flush=True)

  mmat = HDF5Matrix(massmat_path, '/massmat')
  jac = HDF5Matrix(jacobian_path, '/jacobian')

  mmat_big = sparse.block_diag(mass_blocks(mmat.csr_rep, gamma), format='csr')
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
  real_jac = (solved - identity) / timestep

  _save_cached(cache_dir, REAL_JACOBIAN_CACHE, real_jac)
  return real_jac


def spectrum_from_schur(
  schur_t: NDArray[np.complexfloating],
) -> NDArray[np.complex128]:
  """The spectrum, which is already sitting on the diagonal of `T`.

  Not persisted here; `run` saves it as `pseudo_eigvals.npy` beside the samples.
  """
  return np.asarray(schur_t.diagonal(), dtype=np.complex128)


def rightmost_indices(
  eigvals: NDArray[np.complexfloating],
  n: int,
) -> NDArray[np.intp]:
  """Diagonal positions of the `n` rightmost eigenvalues, by real part alone.

  A conjugate pair therefore yields two entries with identical `Re(v)`.
  """
  return np.asarray(np.argsort(-eigvals.real, kind='stable')[:n], dtype=np.intp)


#: Back-substitution rescales whenever a component grows past this.
_GROWTH_LIMIT = 1e150


def eigenvectors_from_schur(
  schur_t: NDArray[np.complexfloating],
  schur_z: NDArray[np.complexfloating],
  indices: NDArray[np.intp],
) -> NDArray[np.complex128]:
  """Right eigenvectors at the given diagonal positions, in the physical basis.

  `T` upper triangular, so `T[i,i]`'s vector back-substitutes; `Z y` maps back.
  """
  # y[i] = 1, zeros above i, rest from (T[:i,:i] - lambda I) y = -T[:i,i].
  if schur_t.shape != schur_z.shape:
    raise ValueError(
      f'Schur factor {schur_t.shape} and vectors {schur_z.shape} disagree')

  # Floor on the denominator, as LAPACK trevc uses for repeated eigenvalues.
  smin = float(np.finfo(np.float64).eps * np.abs(schur_t.diagonal()).max())
  out = np.empty((schur_t.shape[0], indices.size), dtype=np.complex128)

  for col, i in enumerate(int(k) for k in indices):
    lam = schur_t[i, i]
    y = np.zeros(i + 1, dtype=np.complex128)
    y[i] = 1.0
    for j in range(i - 1, -1, -1):
      denom = schur_t[j, j] - lam
      if abs(denom) < smin:
        denom = complex(smin)
      y[j] = -(schur_t[j, j + 1:i + 1] @ y[j + 1:]) / denom
      if abs(y[j]) > _GROWTH_LIMIT:
        y /= abs(y[j])
    vec = schur_z[:, :i + 1] @ y
    out[:, col] = vec / np.linalg.norm(vec)

  return out


def write_eigenmode_restarts(
  schur_t: NDArray[np.complexfloating],
  schur_z: NDArray[np.complexfloating],
  keep_global: NDArray[np.bool_],
  nr_local: int,
  output_dir: str,
  n_eigvecs: int = 40,
) -> tuple[NDArray[np.complex128], list[str]]:
  """Write the rightmost eigenvectors as restarts; returns eigenvalues and paths.

  The caller pairs the two into `eigenmodes.json`; `io` cannot be imported here.
  """
  eigvals = spectrum_from_schur(schur_t)
  indices = rightmost_indices(eigvals, n_eigvecs)
  chosen = eigvals[indices]
  print(f'extracting {indices.size} eigenvectors from the Schur factor, '
        f'rightmost Re lambda = {chosen.real.max():.6g}', flush=True)

  eigvecs = eigenvectors_from_schur(schur_t, schur_z, indices)
  paths = write_restart_eigenvectors(
    eigvecs, keep_global, nr_local, os.path.join(output_dir, 'eigvecs_plot'))
  return chosen, paths


def _compute_schur(
  real_jac: NDArray[np.float64],
  cache_dir: str,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
  """Factorise A = Z T Z*, caching both halves and plotting the diagonal."""
  print('computing schur factorization', flush=True)
  schur_raw, vectors_raw = scipy.linalg.schur(real_jac, output='complex')
  schur_t = np.asarray(schur_raw, dtype=np.complex128)
  schur_z = np.asarray(vectors_raw, dtype=np.complex128)

  os.makedirs(cache_dir, exist_ok=True)
  spectrum = spectrum_from_schur(schur_t)
  plt.figure()
  plt.scatter(spectrum.real, spectrum.imag, s=2, c='k')
  plt.savefig(os.path.join(cache_dir, SCHUR_PLOT))
  plt.close()

  _save_cached(cache_dir, SCHUR_CACHE, schur_t)
  # Saved unconditionally; Z cannot be recovered from T alone.
  _save_cached(cache_dir, SCHURVEC_CACHE, schur_z)
  return schur_t, schur_z


def load_or_compute_schur(
  real_jac: NDArray[np.float64],
  cache_dir: str = DEFAULT_CACHE_DIR,
) -> NDArray[np.complex128]:
  """Load the cached Schur factor T, or compute and persist it.

  The cheap entry point for anything that only samples, so `Z` is never loaded.
  """
  cached = _load_cached(cache_dir, SCHUR_CACHE, 'Schur factor')
  if cached is not None:
    return np.asarray(cached, dtype=np.complex128)
  schur_t, _ = _compute_schur(real_jac, cache_dir)
  return schur_t


def load_or_compute_schur_vectors(
  real_jac: NDArray[np.float64],
  cache_dir: str = DEFAULT_CACHE_DIR,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
  """Load or compute both halves of A = Z T Z*."""
  # Both halves must be cached to skip the factorisation; T alone redoes it.
  cached_t = _load_cached(cache_dir, SCHUR_CACHE, 'Schur factor')
  cached_z = _load_cached(cache_dir, SCHURVEC_CACHE, 'Schur vectors')
  if cached_t is not None and cached_z is not None:
    return (np.asarray(cached_t, dtype=np.complex128),
            np.asarray(cached_z, dtype=np.complex128))

  if cached_t is not None:
    print(
      f'{SCHUR_CACHE} is cached but {SCHURVEC_CACHE} is not; Z cannot be '
      f'recovered from T alone, so the factorisation is being redone. It will '
      f'be cached afterwards.', flush=True)
  return _compute_schur(real_jac, cache_dir)
