"""Pseudospectrum sampling: sigma_min(zI - T) over a complex grid.

Sampling is parallelised with a `fork` pool. The Schur factor is published to
module globals in the parent so workers inherit it through the fork rather than
pickling a dense matrix per task.
"""

import multiprocessing
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy import sparse

from .grid import _grid_shape, _is_real_axis_symmetric_grid

#: Inherited by fork workers; set by `_compute_sigmin_points` in the parent.
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
  vals, _ = sparse.linalg.eigsh(op, k=1, which='LM', ncv=20, tol=1e-6)
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


def _compute_sigmin_points(
  zz_flat: NDArray[np.complex128],
  T: NDArray[np.complexfloating],
  nprocs: int,
  progress_label: str = 'pseudospectrum points',
) -> NDArray[np.float64]:
  """Evaluate sigma_min at flattened complex sample points in parallel."""
  zz_flat = np.asarray(zz_flat).ravel()
  sigmin_flat = np.zeros((zz_flat.shape[0],))
  worker_count = _worker_count(zz_flat.shape[0], nprocs)
  print(
    f"using nprocs={worker_count}, work_items={zz_flat.shape[0]}",
    flush=True)

  global _worker_T, _worker_trtrs
  _worker_T = T
  _worker_trtrs = None
  try:
    ctx = multiprocessing.get_context('fork')
    pool = ctx.Pool(
      processes=worker_count,
      initializer=_init_worker_from_parent)
    completed = 0
    try:
      for idx, sig_val in pool.imap_unordered(_compute_sig_point, enumerate(zz_flat), chunksize=1):
        sigmin_flat[idx] = sig_val
        completed += 1
        if completed % worker_count == 0 or completed == zz_flat.shape[0]:
          print(f'completed {completed}/{zz_flat.shape[0]} {progress_label}', flush=True)
      print('all pseudospectrum tasks completed; closing worker pool', flush=True)
      pool.close()
      pool.join()
    except Exception:
      pool.terminate()
      pool.join()
      raise
  finally:
    _worker_T = None
    _worker_trtrs = None

  return sigmin_flat


def compute_pseudospectrum(
  imat: NDArray[np.complexfloating],
  grid_points: int = 128,
  nprocs: int = 10,
  real_min: float | None = None,
  real_max: float | None = None,
  imag_min: float | None = None,
  imag_max: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """Compute the pseudospectrum over a rectangular complex grid.

  If the sampled imaginary axis is symmetric about zero, only the upper
  half-plane is evaluated and then mirrored to the lower half-plane.

  That mirroring is only valid when the operator's pseudospectrum is symmetric
  about the real axis, i.e. when its spectrum is closed under conjugation. This
  holds for the real reduced operator built by `operator.py`, but a complex
  `imat` with a non-conjugate-symmetric spectrum will produce a silently wrong
  lower half-plane. Pass asymmetric imaginary bounds to force full-grid
  evaluation in that case.
  """
  T = imat
  if real_min is None or real_max is None or imag_min is None or imag_max is None:
    raise ValueError('real/imag bounds must all be provided')

  if real_max <= real_min:
    raise ValueError('real_max must be greater than real_min')
  if imag_max <= imag_min:
    raise ValueError('imag_max must be greater than imag_min')

  nrows, ncols = _grid_shape(grid_points)
  r = np.linspace(real_min, real_max, ncols)
  c = np.linspace(imag_min, imag_max, nrows)

  R, C = np.meshgrid(r, c)
  print(
    f"computing pseudospectrum on Re[z] in [{real_min:.6g}, {real_max:.6g}] "
    f"and Im[z] in [{imag_min:.6g}, {imag_max:.6g}] with shape=({nrows},{ncols})",
    flush=True)

  zz = R + 1j * C
  total_points = int(zz.size)
  if _is_real_axis_symmetric_grid(c):
    upper_start = nrows // 2
    upper_rows = np.arange(upper_start, nrows, dtype=int)
    zz_upper_flat = zz[upper_rows, :].ravel()
    print(
      f'using real-axis symmetry: evaluating upper half only '
      f'({zz_upper_flat.shape[0]}/{total_points} points)',
      flush=True)
    sigmin_upper_flat = _compute_sigmin_points(
      zz_upper_flat,
      T,
      nprocs,
      progress_label='upper-half pseudospectrum points')
    sigmin_upper = sigmin_upper_flat.reshape(upper_rows.shape[0], ncols)
    sigmin = np.zeros((nrows, ncols))
    sigmin[upper_rows, :] = sigmin_upper
    mirror_rows = nrows - 1 - upper_rows
    sigmin[mirror_rows, :] = sigmin_upper
  else:
    print(
      'imaginary-axis grid is not symmetric about zero; computing full grid',
      flush=True)
    sigmin_flat = _compute_sigmin_points(
      zz.ravel(),
      T,
      nprocs,
      progress_label='pseudospectrum points')
    sigmin = sigmin_flat.reshape(nrows, ncols)

  return R, C, sigmin


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
