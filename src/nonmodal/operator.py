"""Build the reduced continuous-time generator and its spectral factorisations.

OFT's backward-Euler Jacobian is exactly `J = M + dt*K`, so the generator of
`du/dt = L u` is `L = M^-1 (M - J) / dt`, independent of `dt`. Its eigenvalues
are physical growth rates in 1/s.

Each stage caches to `.npy` and reuses it blindly. The Schur caches of a
weighted norm carry its tag (`generator_schur_l2s.npy`); `generator.npy`
holds `L` itself and is shared by every norm. Caches are keyed by filename
only, so they are NOT invalidated when inputs or `KEPT_BLOCK_IDS` change --
delete them by hand.

* `generator_from_matrices` -- `L` from a reduced Jacobian and mass matrix.
* `load_or_compute_generator` -- the reduced generator `L`, from the exports.
* `load_or_compute_schur` -- the triangular factor `T` of `L = Z T Z*`, or
  of `B = R L R^-1` in a weighted norm (see `norm.py`).
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

from .adapters import find_input_deck, read_adiabatic_index, read_resistivity, read_timestep
from .fields import (
  FIELD_BLOCK_COUNT,
  PSI_BLOCK_ID,
  TEMPERATURE_BLOCK_ID,
  write_restart_eigenvectors,
)
from .matrices import HDF5Matrix, assemble_reduced
from .norm import L2_COEFFICIENTS, Norm

GENERATOR_CACHE = 'generator.npy'
SCHUR_CACHE = 'generator_schur.npy'
#: The unitary Z of L = Z T Z*, needed to map a pseudomode back to the basis.
SCHURVEC_CACHE = 'generator_schurvecs.npy'
SCHUR_PLOT = 'generator_eigs.png'
#: Caches of the old backward-Euler operator (J^-1 M - I)/dt. Never read: they
#: hold a different operator, and caches are keyed by filename alone.
LEGACY_CACHES = ('real_jacobian.npy', 'full_reduced_schur.npy', 'full_reduced_schurvecs.npy')
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


def mass_blocks(scalar_mass: Any, gamma: float, psi_scale: float) -> list[Any]:
  """The seven field blocks of the global mass matrix, given the scalar one.

  Temperature carries `1/(gamma - 1)` and psi carries `psi_scale`; the rest
  carry 1. These match the row scalings of OFT's own mass function.
  """
  # The export is the scalar Lagrange mass matrix, so no per-field factor can
  # be present in it; OFT applies these when it assembles its own operator.
  blocks = [scalar_mass] * FIELD_BLOCK_COUNT
  blocks[TEMPERATURE_BLOCK_ID] = scalar_mass / (gamma - 1.0)
  blocks[PSI_BLOCK_ID] = scalar_mass * psi_scale
  return blocks


#: How far the measured psi ratio may sit from a candidate scale, as a factor.
PSI_SCALE_TOLERANCE = 1.5


def psi_mass_scale(
  local_jacobian: Any,
  bc_flags: NDArray[np.bool_],
  scalar_mass: Any,
  eta: float,
) -> float:
  """The psi mass scale an export needs: 1, or 1/eta for upstream OFT.

  Upstream OFT (commit 5da9985) divides the psi rows of both its mass function
  and its Jacobian by eta; older builds do not. The Jacobian therefore records
  which build made it: its psi-psi block is the mass matrix times that scale,
  plus a small timestep-weighted term. The ratio is measured on interior rows
  of the local matrices and must land near one candidate, or this raises.
  """
  nb = scalar_mass.shape[0]
  rows = slice(PSI_BLOCK_ID * nb, (PSI_BLOCK_ID + 1) * nb)
  interior = ~np.asarray(bc_flags[rows], dtype=bool)
  jac_psi = sparse.csr_array(local_jacobian)[rows, rows][interior]
  mass = sparse.csr_array(scalar_mass)[interior]
  ratio = float(jac_psi.multiply(mass).sum() / mass.multiply(mass).sum())

  candidates = {'unscaled (pre-5da9985 OFT)': 1.0, 'scaled by 1/eta (upstream OFT)': 1.0 / eta}
  label, scale = min(candidates.items(), key=lambda item: abs(np.log(ratio / item[1])))
  if not (ratio > 0 and abs(np.log(ratio / scale)) <= np.log(PSI_SCALE_TOLERANCE)):
    raise ValueError(
      f'psi rows of the Jacobian are {ratio:.4g} x the mass matrix, near neither '
      f'1 nor 1/eta = {1.0 / eta:.4g}; the export and deck do not match')
  print(f'psi mass block {label}: Jacobian/mass ratio {ratio:.4g}, '
        f'scale {scale:.6g}', flush=True)
  return scale


def generator_from_matrices(
  reduced_jac: Any,
  reduced_mmat: Any,
  timestep: float,
) -> NDArray[np.float64]:
  """The generator `L = M^-1 (M - J) / dt` of a backward-Euler Jacobian `J = M + dt*K`.

  This is `-M^-1 K`, the continuous-time operator, and does not depend on `dt`.
  `timestep` must be the one the Jacobian was exported with.
  """
  difference = sparse.csr_array(reduced_mmat - reduced_jac).toarray()
  solved = sparse.linalg.spsolve(sparse.csc_array(reduced_mmat), difference)
  return np.asarray(solved, dtype=np.float64) / timestep


def check_timestep(timestep: float, deck_timestep: float, deck: str) -> None:
  """Raise unless `timestep` is the `dt` the export was made with."""
  if not np.isclose(timestep, deck_timestep, rtol=1e-12, atol=0.0):
    raise ValueError(
      f'--timestep {timestep:g} differs from dt={deck_timestep:g} in {deck}; the '
      f'Jacobian is M + dt*K at the export dt, so any other value yields a wrong L')


def _warn_legacy_caches(cache_dir: str) -> None:
  """Say so when old-operator caches sit in the cache directory, unused."""
  present = [name for name in LEGACY_CACHES if os.path.exists(os.path.join(cache_dir, name))]
  if present:
    print(f'ignoring {", ".join(present)} in {os.path.abspath(cache_dir)}: they hold '
          f'the old operator (J^-1 M - I)/dt, not the generator L', flush=True)


def load_or_compute_generator(
  jacobian_path: str,
  massmat_path: str,
  keep_global: NDArray[np.bool_],
  cache_dir: str = DEFAULT_CACHE_DIR,
  timestep: float = DEFAULT_TIMESTEP,
) -> NDArray[np.float64]:
  """Load the cached reduced generator `L`, or build and cache it from HDF5 matrices."""
  _warn_legacy_caches(cache_dir)
  cached = _load_cached(cache_dir, GENERATOR_CACHE, 'reduced generator')
  if cached is not None:
    return cached

  # gamma, eta and dt define the operator, so they come from the deck that ran the export.
  deck = find_input_deck(jacobian_path)
  gamma = read_adiabatic_index(deck)
  eta = read_resistivity(deck)
  check_timestep(timestep, read_timestep(deck), deck)
  print(f'input deck {deck}: gamma={gamma:g}, eta={eta:g}, dt={timestep:g}', flush=True)
  print(f'temperature mass block scaled by 1/(gamma-1), gamma={gamma:g}', flush=True)

  mmat = HDF5Matrix(massmat_path, '/massmat')
  jac = HDF5Matrix(jacobian_path, '/jacobian')
  psi_scale = psi_mass_scale(jac.csr_rep, jac.bc_flags, mmat.csr_rep, eta)

  mmat_big = sparse.block_diag(mass_blocks(mmat.csr_rep, gamma, psi_scale), format='csr')
  del mmat

  reduced_jac = assemble_reduced(jac.csr_rep, jac, keep_global)
  reduced_mmat = assemble_reduced(mmat_big, jac, keep_global)
  del mmat_big

  print('Shape of reduced_Jac is:', reduced_jac.shape, flush=True)
  print('shape of reduced_Mmat is:', reduced_mmat.shape, flush=True)

  generator = generator_from_matrices(reduced_jac, reduced_mmat, timestep)
  del reduced_jac, reduced_mmat

  _save_cached(cache_dir, GENERATOR_CACHE, generator)
  return generator


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
  # Z may be tall: n x k for T restricted to a k-dimensional invariant subspace.
  if schur_t.shape[0] != schur_t.shape[1] or schur_z.shape[1] != schur_t.shape[0]:
    raise ValueError(
      f'Schur factor {schur_t.shape} and vectors {schur_z.shape} disagree')

  # Floor on the denominator, as LAPACK trevc uses for repeated eigenvalues.
  smin = float(np.finfo(np.float64).eps * np.abs(schur_t.diagonal()).max())
  out = np.empty((schur_z.shape[0], indices.size), dtype=np.complex128)

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
  norm: Norm = L2_COEFFICIENTS,
) -> tuple[NDArray[np.complex128], list[str]]:
  """Write the rightmost eigenvectors as restarts; returns eigenvalues and paths.

  `schur_t`, `schur_z` factor the generator in `norm`'s coordinates, so each
  vector is mapped back to coefficients before it is written.
  The caller pairs the two into `eigenmodes.json`; `io` cannot be imported here.
  """
  eigvals = spectrum_from_schur(schur_t)
  indices = rightmost_indices(eigvals, n_eigvecs)
  chosen = eigvals[indices]
  print(f'extracting {indices.size} eigenvectors from the Schur factor, '
        f'rightmost Re lambda = {chosen.real.max():.6g}', flush=True)

  eigvecs = norm.to_basis(eigenvectors_from_schur(schur_t, schur_z, indices))
  eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)
  paths = write_restart_eigenvectors(
    eigvecs, keep_global, nr_local, os.path.join(output_dir, 'eigvecs_plot'))
  return chosen, paths


def _compute_schur(
  generator: NDArray[np.float64],
  cache_dir: str,
  norm: Norm,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
  """Factorise the generator in `norm`'s coordinates, caching both halves."""
  operator = norm.operator(generator)
  print(f'computing schur factorization ({norm.name} norm)', flush=True)
  schur_raw, vectors_raw = scipy.linalg.schur(operator, output='complex')
  del operator
  schur_t = np.asarray(schur_raw, dtype=np.complex128)
  schur_z = np.asarray(vectors_raw, dtype=np.complex128)

  os.makedirs(cache_dir, exist_ok=True)
  spectrum = spectrum_from_schur(schur_t)
  plt.figure()
  plt.scatter(spectrum.real, spectrum.imag, s=2, c='k')
  plt.savefig(os.path.join(cache_dir, norm.cache_name(SCHUR_PLOT)))
  plt.close()

  _save_cached(cache_dir, norm.cache_name(SCHUR_CACHE), schur_t)
  # Saved unconditionally; Z cannot be recovered from T alone.
  _save_cached(cache_dir, norm.cache_name(SCHURVEC_CACHE), schur_z)
  return schur_t, schur_z


def load_or_compute_schur(
  generator: NDArray[np.float64],
  cache_dir: str = DEFAULT_CACHE_DIR,
  norm: Norm = L2_COEFFICIENTS,
) -> NDArray[np.complex128]:
  """Load the cached Schur factor T in `norm`, or compute and persist it.

  The cheap entry point for anything that only samples, so `Z` is never loaded.
  """
  cached = _load_cached(cache_dir, norm.cache_name(SCHUR_CACHE), f'Schur factor ({norm.name})')
  if cached is not None:
    return np.asarray(cached, dtype=np.complex128)
  schur_t, _ = _compute_schur(generator, cache_dir, norm)
  return schur_t


def load_or_compute_schur_vectors(
  generator: NDArray[np.float64],
  cache_dir: str = DEFAULT_CACHE_DIR,
  norm: Norm = L2_COEFFICIENTS,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
  """Load or compute both halves of the generator's Schur form in `norm`."""
  schur_name, vector_name = norm.cache_name(SCHUR_CACHE), norm.cache_name(SCHURVEC_CACHE)
  # Both halves must be cached to skip the factorisation; T alone redoes it.
  cached_t = _load_cached(cache_dir, schur_name, f'Schur factor ({norm.name})')
  cached_z = _load_cached(cache_dir, vector_name, f'Schur vectors ({norm.name})')
  if cached_t is not None and cached_z is not None:
    return (np.asarray(cached_t, dtype=np.complex128),
            np.asarray(cached_z, dtype=np.complex128))

  if cached_t is not None:
    print(
      f'{schur_name} is cached but {vector_name} is not; Z cannot be '
      f'recovered from T alone, so the factorisation is being redone. It will '
      f'be cached afterwards.', flush=True)
  return _compute_schur(generator, cache_dir, norm)
