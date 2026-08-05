"""Pseudospectrum sampling: sigma_min(zI - T) at arbitrary complex points.

Sampling is parallelised with a `fork` pool. The Schur factor is published to
module globals in the parent so workers inherit it through the fork rather than
pickling a dense matrix per task.

There is one primitive, `sample_sigmin`, which evaluates a flat array of points.
Everything else -- uniform grids, adaptive refinement, half-plane mirroring --
is a question of *which* points to hand it, and lives in sampling.py/refine.py.
"""

import functools
import multiprocessing
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy import sparse

from .sampling import Bounds, near_square, uniform_points

#: Inherited by fork workers; set by `sample_sigmin` in the parent.
_worker_T: NDArray[np.complexfloating] | None = None
_worker_trtrs: Callable[..., Any] | None = None


def _init_worker_from_parent() -> None:
  """Initialise worker-local LAPACK function pointers."""
  global _worker_T, _worker_trtrs
  if _worker_T is None:
    raise RuntimeError('worker did not inherit operator matrix')
  # get_lapack_funcs returns one callable per requested name, but scipy-stubs
  # types the result as list-or-single, so narrow it explicitly.
  funcs = scipy.linalg.get_lapack_funcs(('trtrs',), (_worker_T,))
  _worker_trtrs = cast(Callable[..., Any], funcs[0] if isinstance(funcs, list) else funcs)


#: Seed for the deterministic ARPACK start vector. Any fixed vector works; a
#: pseudo-random one avoids the accidental orthogonality an all-ones vector can
#: hit on structured operators.
_START_VECTOR_SEED = 20260805


@functools.cache
def _start_vector(n: int) -> NDArray[np.float64]:
  """A fixed ARPACK starting vector of length n, cached per size."""
  return np.random.default_rng(_START_VECTOR_SEED).standard_normal(n)


def _compute_sig_for_z_from_factors(
  z: complex,
  T: NDArray[np.complexfloating],
  trtrs: Callable[..., Any],
) -> float:
  """Compute sigma_min(zI - T) using triangular solves on Schur factors."""
  # Avoid materializing z*I, which creates an extra dense allocation.
  T1 = -T.copy()
  T1.flat[::T1.shape[0] + 1] += z

  def _matvec(q: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    tmp, _ = trtrs(T1, q.reshape(-1, 1), lower=0, trans=2, unitdiag=0)
    result, _ = trtrs(T1, tmp, lower=0, trans=0, unitdiag=0)
    return result.ravel()

  op = sparse.linalg.LinearOperator(
    T1.shape,
    matvec=_matvec,
    dtype=np.complex128)
  # A fixed starting vector makes runs reproducible. ARPACK otherwise draws one
  # at random, so repeating a run gave slightly different values -- harmless in
  # aggregate but it makes results impossible to reproduce exactly, and on an
  # ill-conditioned operator the spread reaches ~1e-4 relative.
  vals, _ = sparse.linalg.eigsh(
    op, k=1, which='LM', ncv=20, tol=1e-6, v0=_start_vector(T1.shape[0]))
  sig_min = vals[0]
  return 1 / np.sqrt(sig_min)


def _compute_sig_point(item: tuple[int, complex]) -> tuple[int, float]:
  """Worker entry point for a single complex grid point."""
  idx, z = item
  if _worker_T is None or _worker_trtrs is None:
    raise RuntimeError('worker operator state is not initialized')
  return idx, _compute_sig_for_z_from_factors(z, _worker_T, _worker_trtrs)


def _worker_count(total_points: int, requested_nprocs: int) -> int:
  """Use as many workers as requested, up to one worker per work item."""
  return max(1, min(int(requested_nprocs), int(total_points)))


def sample_sigmin(
  points: NDArray[np.complex128],
  T: NDArray[np.complexfloating],
  nprocs: int,
  progress_label: str = 'pseudospectrum points',
) -> NDArray[np.float64]:
  """Evaluate sigma_min(zI - T) at each of `points`, in parallel."""
  points = np.asarray(points).ravel()
  sigmin = np.zeros((points.shape[0],))
  if points.size == 0:
    return sigmin
  worker_count = _worker_count(points.shape[0], nprocs)
  print(f'using nprocs={worker_count}, work_items={points.shape[0]}', flush=True)

  global _worker_T, _worker_trtrs
  _worker_T = T
  _worker_trtrs = None
  try:
    ctx = multiprocessing.get_context('fork')
    pool = ctx.Pool(processes=worker_count, initializer=_init_worker_from_parent)
    completed = 0
    try:
      for idx, sig_val in pool.imap_unordered(
        _compute_sig_point, enumerate(points), chunksize=1
      ):
        sigmin[idx] = sig_val
        completed += 1
        if completed % worker_count == 0 or completed == points.shape[0]:
          print(f'completed {completed}/{points.shape[0]} {progress_label}', flush=True)
      pool.close()
      pool.join()
    except Exception:
      pool.terminate()
      pool.join()
      raise
  finally:
    _worker_T = None
    _worker_trtrs = None

  return sigmin


def compute_pseudospectrum(
  imat: NDArray[np.complexfloating],
  grid_points: int = 128,
  nprocs: int = 10,
  real_min: float | None = None,
  real_max: float | None = None,
  imag_min: float | None = None,
  imag_max: float | None = None,
  nx: int | None = None,
  ny: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """Sample a uniform rectangular grid and return it as a mesh.

  A convenience wrapper over `sample_sigmin` for the common rectangular case,
  returning `(R, C, sigmin)` meshes ready for contouring. `grid_points` is
  split into a near-square `(nx, ny)`; pass `nx`/`ny` to control the shape
  directly.

  `imat` must already be in Schur (upper-triangular) form.
  """
  if real_min is None or real_max is None or imag_min is None or imag_max is None:
    raise ValueError('real/imag bounds must all be provided')
  bounds = Bounds(real_min, real_max, imag_min, imag_max)

  if nx is None or ny is None:
    nx, ny = near_square(grid_points)

  points = uniform_points(bounds, nx, ny)
  print(
    f'computing pseudospectrum on Re[z] in [{real_min:.6g}, {real_max:.6g}] '
    f'and Im[z] in [{imag_min:.6g}, {imag_max:.6g}] with shape=({ny},{nx})',
    flush=True)

  sigmin = sample_sigmin(points, imat, nprocs)
  R = points.real.reshape(ny, nx)
  C = points.imag.reshape(ny, nx)
  return R, C, sigmin.reshape(ny, nx)


def choose_contour_levels(
  sigmin: NDArray[np.float64],
  min_level: float = 1e-7,
  nlevels: int = 5,
) -> NDArray[np.float64]:
  """Choose positive contour levels spanning available pseudospectrum values."""
  if nlevels < 1:
    raise ValueError('nlevels must be >= 1')
  if min_level <= 0:
    raise ValueError('min_level must be positive')

  vals = np.asarray(sigmin)
  mask = np.isfinite(vals) & (vals > 0)
  if not np.any(mask):
    raise ValueError('sigmin has no finite positive entries')

  data_min = float(np.min(vals[mask]))
  data_max = float(np.max(vals[mask]))

  # Use geometric spacing across finite data. If the requested minimum level
  # exceeds data_max, fall back to full data range so contours remain meaningful.
  if min_level < data_max:
    lo = max(min_level, data_min)
  else:
    lo = data_min
  hi = data_max
  if hi <= lo:
    # Keep at least two boundaries for degenerate/near-constant fields.
    return np.array([lo, np.nextafter(lo, np.inf)])
  return np.geomspace(lo, hi, nlevels)
