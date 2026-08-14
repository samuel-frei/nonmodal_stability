"""Pseudospectrum sampling: sigma_min(zI - T) at arbitrary complex points.

One primitive, `sample_sigmin`, evaluates a flat array of points; everything
else is a question of *which* points to hand it (sampling.py, refine.py). It is
parallelised with a `fork` pool, the Schur factor reaching workers through
inherited module globals rather than being pickled per task.

* `sample_sigmin` -- evaluate a flat point set in parallel.
* `sigmin_with_mode` -- one point, keeping the eigenvector as well.
* `compute_pseudospectrum` -- convenience wrapper returning `(R, C, sigmin)`.
* `choose_contour_levels` -- geometric levels spanning the sampled values.
"""

import functools
import multiprocessing
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy import sparse

from .sampling import Bounds, uniform_points

#: Inherited by fork workers; set by `sample_sigmin` in the parent.
_worker_T: NDArray[np.complexfloating] | None = None
_worker_trtrs: Callable[..., Any] | None = None


def _init_worker_from_parent() -> None:
  """Initialise worker-local LAPACK function pointers."""
  global _worker_T, _worker_trtrs
  if _worker_T is None:
    raise RuntimeError('worker did not inherit operator matrix')
  # scipy-stubs types get_lapack_funcs as list-or-single, so narrow it.
  funcs = scipy.linalg.get_lapack_funcs(('trtrs',), (_worker_T,))
  _worker_trtrs = cast(Callable[..., Any], funcs[0] if isinstance(funcs, list) else funcs)


#: Seeds the fixed, pseudo-random ARPACK start vector.
_START_VECTOR_SEED = 20260805


@functools.cache
def _start_vector(n: int) -> NDArray[np.float64]:
  """A fixed ARPACK starting vector of length n, cached per size."""
  return np.random.default_rng(_START_VECTOR_SEED).standard_normal(n)


def _shifted_resolvent_operator(
  z: complex,
  T: NDArray[np.complexfloating],
  trtrs: Callable[..., Any],
) -> sparse.linalg.LinearOperator:
  """`((zI - T)* (zI - T))^-1`, applied through triangular solves."""
  # Wright & Trefethen (2001) §2, in two stages of triangular solves: M* y = q
  # then M x = y, giving (M* M)^-1, whose eigenvector is the right singular
  # vector -- the pseudomode. The shift is written onto the diagonal in place.
  T1 = -T.copy()
  T1.flat[::T1.shape[0] + 1] += z

  def _matvec(q: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    tmp, _ = trtrs(T1, q.reshape(-1, 1), lower=0, trans=2, unitdiag=0)
    result, _ = trtrs(T1, tmp, lower=0, trans=0, unitdiag=0)
    return result.ravel()

  return sparse.linalg.LinearOperator(T1.shape, matvec=_matvec, dtype=np.complex128)


def _compute_sig_for_z_from_factors(
  z: complex,
  T: NDArray[np.complexfloating],
  trtrs: Callable[..., Any],
) -> float:
  """Compute sigma_min(zI - T) using triangular solves on Schur factors."""
  op = _shifted_resolvent_operator(z, T, trtrs)
  # The fixed start vector makes repeated runs bitwise identical.
  vals, _ = sparse.linalg.eigsh(
    op, k=1, which='LM', ncv=20, tol=1e-6, v0=_start_vector(T.shape[0]))
  sig_min = vals[0]
  return 1 / np.sqrt(sig_min)


#: Tighter than the sampler's 1e-6: eigenvectors converge more slowly.
DEFAULT_MODE_TOL = 1e-8


def sigmin_with_mode(
  z: complex,
  T: NDArray[np.complexfloating],
  tol: float = DEFAULT_MODE_TOL,
) -> tuple[float, NDArray[np.complex128]]:
  """`sigma_min(zI - T)` and its unit right singular vector, in the Schur basis.

  The sampler's iteration, keeping the eigenvector it discards. See `pseudomode`.
  """
  funcs = scipy.linalg.get_lapack_funcs(('trtrs',), (T,))
  trtrs = cast(Callable[..., Any], funcs[0] if isinstance(funcs, list) else funcs)

  op = _shifted_resolvent_operator(z, T, trtrs)
  n = int(T.shape[0])
  vals, vecs = sparse.linalg.eigsh(
    op, k=1, which='LM', ncv=min(n, 20), tol=tol, v0=_start_vector(n))

  mode = np.asarray(vecs[:, 0], dtype=np.complex128)
  return float(1.0 / np.sqrt(float(vals[0]))), mode / np.linalg.norm(mode)


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
  bounds: Bounds,
  nx: int,
  ny: int,
  nprocs: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """Sample a uniform `nx` by `ny` grid over `bounds`, returned as meshes.

  For callers wanting a fixed lattice; `imat` must already be triangular.
  """
  points = uniform_points(bounds, nx, ny)
  print(
    f'computing pseudospectrum on Re[z] in '
    f'[{bounds.real_min:.6g}, {bounds.real_max:.6g}] and Im[z] in '
    f'[{bounds.imag_min:.6g}, {bounds.imag_max:.6g}] with shape=({ny},{nx})',
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

  # Geometric spacing across the finite data, from min_level where it fits.
  if min_level < data_max:
    lo = max(min_level, data_min)
  else:
    lo = data_min
  hi = data_max
  if hi <= lo:
    # A near-constant field still gets two distinct boundaries.
    return np.array([lo, np.nextafter(lo, np.inf)])
  return np.geomspace(lo, hi, nlevels)
