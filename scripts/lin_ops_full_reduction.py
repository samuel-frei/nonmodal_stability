"""Compute reduced-operator pseudospectra and export interactive plots.

This script loads OFT Jacobian and mass matrices, reduces the global system,
builds the Schur form of the reduced operator, samples the pseudospectrum over
user-defined complex-plane bounds, and writes both arrays and interactive HTML
plots. It is designed for batch execution and supports multiprocessing.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import multiprocessing
import socket
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
from numba import njit
from scipy import sparse


FIELD_BLOCK_COUNT = 7
KEPT_BLOCK_IDS = (0, 1, 3, 4, 5)
REAL_JACOBIAN_CACHE = './real_jacobian.npy'
EIGVAL_CACHE = './full_reduced_eigvals.npy'
EIGVEC_CACHE = './full_reduced_eigvecs.npy'
SCHUR_CACHE = './full_reduced_schur.npy'
DT = 1e-7


class Matrix:
  """Matrix wrapper for OFT HDF5 matrix datasets."""

  def __init__(self, filename: str, mat_name: str):
    with h5py.File(filename, 'r') as f:
      self.mat_name = mat_name
      self.nr = f[f'{mat_name}/nr'][0]
      self.nc = f[f'{mat_name}/nc'][0]
      self.nrg = f[f'{mat_name}/nrg'][0]
      self.ncg = f[f'{mat_name}/ncg'][0]
      self.lc = np.array(f[f'{mat_name}/lc'])-1
      self.lg = np.array(f[f'{mat_name}/lg'])-1
      self.kr = np.array(f[f'{mat_name}/kr'])-1
      self.M = np.array(f[f'{mat_name}/M'])
      try:
        self.bc_flags = np.array(f[f'{mat_name}/bc_flags']) > 0
        self.bcg = np.zeros((self.nrg,), dtype=bool)
        self.bcg[self.lg] = self.bc_flags
      except KeyError:
        self.bc_flags = np.zeros(self.nr, dtype=bool)
        self.bcg = np.zeros((self.nrg,), dtype=bool)
    self.csr_rep = sparse.csr_array((self.M, self.lc, self.kr))


@njit
def make_global_mat(inmat, nrg=None, ncg=None, lg=None):
  """Assemble dense global matrix from local matrix entries using `lg` mapping."""
  outmat = np.zeros((nrg, ncg))
  for i in range(inmat.shape[0]):
    for j in range(inmat.shape[0]):
      ik = lg[i]
      jl = lg[j]
      outmat[ik, jl] += inmat[i, j]
  return outmat

_worker_T = None
_worker_trtrs = None


def _init_worker_from_parent():
  """Initialize worker-local LAPACK function pointers."""
  global _worker_T, _worker_trtrs
  if _worker_T is None:
    raise RuntimeError('worker did not inherit operator matrix')
  _worker_trtrs, = scipy.linalg.get_lapack_funcs(('trtrs',), (_worker_T,))


def _compute_sig_for_z_from_factors(z, T, trtrs):
  """Compute sigma_min(zI - T) using triangular solves on Schur factors."""
  # Avoid materializing z*I, which creates an extra dense allocation.
  T1 = -T.copy()
  T1.flat[::T1.shape[0] + 1] += z

  def _matvec(q):
    tmp, _ = trtrs(T1, q.reshape(-1, 1), lower=0, trans=2, unitdiag=0)
    result, _ = trtrs(T1, tmp, lower=0, trans=0, unitdiag=0)
    return result.ravel()

  op = sparse.linalg.LinearOperator(
    T1.shape,
    matvec=_matvec,
    dtype=np.complex128)
  vals, _ = sparse.linalg.eigsh(op, k=1, which='LM', ncv=10, tol=1e-6)
  sig_min = vals[0]
  return 1 / np.sqrt(sig_min)


def _compute_sig_point(item):
  """Worker entry point for a single complex grid point."""
  idx, z = item
  if _worker_T is None or _worker_trtrs is None:
    raise RuntimeError('worker operator state is not initialized')
  return idx, _compute_sig_for_z_from_factors(z, _worker_T, _worker_trtrs)


def _effective_worker_count(total_points, requested_nprocs):
  """Use as many workers as requested, up to one worker per work item."""
  return max(1, min(int(requested_nprocs), int(total_points)))


def _grid_shape(total_points):
  """Choose factor pair (rows, cols) that is as square as possible."""
  n = int(total_points)
  rows = int(np.floor(np.sqrt(n)))
  while rows > 1 and (n % rows != 0):
    rows -= 1
  cols = n // rows
  return rows, cols


def _is_real_axis_symmetric_grid(c):
  """Return True when sampled imaginary coordinates are symmetric about 0."""
  c = np.asarray(c, dtype=float)
  scale = max(1.0, float(np.max(np.abs(c))))
  atol = max(1e-12, np.finfo(float).eps * scale * 32.0)
  return bool(np.allclose(c, -c[::-1], rtol=0.0, atol=atol))


def _finite_complex_values(values):
  """Return finite complex values with finite real and imaginary parts."""
  vals = np.asarray(values).ravel()
  if vals.size == 0:
    return vals.astype(np.complex128)
  mask = np.isfinite(vals.real) & np.isfinite(vals.imag)
  return vals[mask]


def _bounds_from_complex_values(values):
  """Compute axis-aligned bounds from finite complex points."""
  vals = _finite_complex_values(values)
  if vals.size == 0:
    raise ValueError('no finite complex values available to determine bounds')
  return (
    float(np.min(vals.real)),
    float(np.max(vals.real)),
    float(np.min(vals.imag)),
    float(np.max(vals.imag)))


def _coarse_grid_points(total_points, coarse_fraction):
  """Choose coarse-pass point budget as a fraction of final requested points."""
  target = int(round(float(total_points) * float(coarse_fraction)))
  target = max(16, target)
  return min(int(total_points), target)


def _coarse_seed_bounds_from_spectrum(eigvals, pad_distance):
  """Build coarse-pass bounds from eigenspectrum with conservative padding."""
  re_min, re_max, im_min, im_max = _bounds_from_complex_values(eigvals)
  span_re = max(float(re_max - re_min), 0.0)
  span_im = max(float(im_max - im_min), 0.0)
  span_ref = max(span_re, span_im, 1e-9)
  coarse_pad = max(3.0 * float(pad_distance), 0.35 * span_ref)
  return (
    float(re_min - coarse_pad),
    float(re_max + coarse_pad),
    float(im_min - coarse_pad),
    float(im_max + coarse_pad))


def _extract_contour_vertices(R, C, sigmin, level):
  """Extract contour vertices at one level from sampled pseudospectrum values."""
  sig = np.asarray(sigmin, dtype=float)
  finite = np.isfinite(sig)
  if not np.any(finite):
    return np.empty((0, 2), dtype=float)

  data_min = float(np.nanmin(sig[finite]))
  data_max = float(np.nanmax(sig[finite]))
  if level < data_min or level > data_max:
    return np.empty((0, 2), dtype=float)

  x_axis = np.asarray(R, dtype=float)
  y_axis = np.asarray(C, dtype=float)
  if x_axis.ndim == 2:
    x_axis = x_axis[0, :]
  if y_axis.ndim == 2:
    y_axis = y_axis[:, 0]

  safe_sig = np.array(sig, copy=True)
  safe_sig[~finite] = np.nan

  fig = plt.figure()
  try:
    ax = fig.add_subplot(111)
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', UserWarning)
      contour_set = ax.contour(x_axis, y_axis, safe_sig, levels=[float(level)])
    segments = []
    if contour_set.allsegs:
      for seg in contour_set.allsegs[0]:
        if seg is not None and seg.shape[0] >= 2:
          segments.append(np.asarray(seg[:, :2], dtype=float))
  finally:
    plt.close(fig)

  if not segments:
    return np.empty((0, 2), dtype=float)

  verts = np.vstack(segments)
  keep = np.isfinite(verts).all(axis=1)
  return verts[keep]


def _compute_adaptive_bounds(contour_vertices, eigvals, pad_distance):
  """Compute final adaptive bounds from contour vertices and eigenspectrum fallback."""
  eig_re_min, eig_re_max, eig_im_min, eig_im_max = _bounds_from_complex_values(eigvals)
  source = 'eigenspectrum_fallback'

  verts = np.asarray(contour_vertices, dtype=float)
  if verts.ndim == 2 and verts.shape[0] > 0 and verts.shape[1] >= 2:
    re_vals = verts[:, 0]
    im_vals = verts[:, 1]
    if np.any(np.isfinite(re_vals)) and np.any(np.isfinite(im_vals)):
      source = 'contour_plus_eigenspectrum_union'
      eig_re_min = min(eig_re_min, float(np.nanmin(re_vals)))
      eig_re_max = max(eig_re_max, float(np.nanmax(re_vals)))
      eig_im_min = min(eig_im_min, float(np.nanmin(im_vals)))
      eig_im_max = max(eig_im_max, float(np.nanmax(im_vals)))

  pad = float(pad_distance)
  re_min = eig_re_min - pad
  re_max = eig_re_max + pad
  im_min = eig_im_min - pad
  im_max = eig_im_max + pad

  if re_max <= re_min:
    center = 0.5 * (re_min + re_max)
    half = max(pad, 1e-9)
    re_min = center - half
    re_max = center + half
  if im_max <= im_min:
    center = 0.5 * (im_min + im_max)
    half = max(pad, 1e-9)
    im_min = center - half
    im_max = center + half

  return (float(re_min), float(re_max), float(im_min), float(im_max), source)


def _build_adaptive_axis(axis_min, axis_max, npoints, focus_values, focus_strength):
  """Build a monotonic axis with higher density near focus values."""
  npoints = int(npoints)
  if npoints < 1:
    raise ValueError('npoints must be >= 1')

  axis_min = float(axis_min)
  axis_max = float(axis_max)
  if axis_max <= axis_min:
    raise ValueError('axis_max must be greater than axis_min')

  if npoints == 1:
    return np.array([0.5 * (axis_min + axis_max)], dtype=float)

  focus = np.asarray(focus_values, dtype=float).ravel()
  focus = focus[np.isfinite(focus)]
  focus = focus[(focus >= axis_min) & (focus <= axis_max)]
  if focus.size == 0 or float(focus_strength) <= 0.0:
    return np.linspace(axis_min, axis_max, npoints)

  span = axis_max - axis_min
  probe_count = max(2048, npoints * 32)
  probe = np.linspace(axis_min, axis_max, probe_count)

  if focus.size > 1:
    sorted_focus = np.sort(focus)
    deltas = np.diff(sorted_focus)
    positive = deltas[deltas > 0]
    if positive.size > 0:
      width = float(np.median(positive)) * 2.0
    else:
      width = span / max(8.0 * npoints, 1.0)
  else:
    width = span / max(8.0 * npoints, 1.0)
  width = max(width, span / max(24.0 * npoints, 1.0), 1e-12)

  density = np.ones_like(probe)
  weight = float(focus_strength)
  for center in focus:
    density += weight * np.exp(-0.5 * ((probe - center) / width) ** 2)

  cdf = np.cumsum(density)
  cdf -= cdf[0]
  if cdf[-1] <= 0.0:
    return np.linspace(axis_min, axis_max, npoints)
  cdf /= cdf[-1]

  targets = np.linspace(0.0, 1.0, npoints)
  axis = np.interp(targets, cdf, probe)
  axis[0] = axis_min
  axis[-1] = axis_max

  axis = np.maximum.accumulate(axis)
  scale = max(1.0, abs(axis_min), abs(axis_max))
  eps = np.finfo(float).eps * scale * 64.0
  for i in range(1, axis.size):
    if axis[i] <= axis[i - 1]:
      axis[i] = axis[i - 1] + eps

  if axis[-1] > axis_max and axis[-1] > axis[0]:
    axis = axis_min + (axis - axis[0]) * ((axis_max - axis_min) / (axis[-1] - axis[0]))
    axis[0] = axis_min
    axis[-1] = axis_max

  return axis


def _build_adaptive_imag_axis(imag_min, imag_max, nrows, focus_values, focus_strength):
  """Build adaptive imaginary axis, preserving exact symmetry when feasible."""
  imag_min = float(imag_min)
  imag_max = float(imag_max)
  nrows = int(nrows)
  if nrows < 1:
    raise ValueError('nrows must be >= 1')

  scale = max(1.0, abs(imag_min), abs(imag_max))
  symmetric = abs(imag_min + imag_max) <= max(1e-12, np.finfo(float).eps * scale * 64.0)
  if not symmetric:
    return _build_adaptive_axis(imag_min, imag_max, nrows, focus_values, focus_strength)

  if nrows == 1:
    return np.array([0.0], dtype=float)

  limit = max(abs(imag_min), abs(imag_max))
  abs_focus = np.abs(np.asarray(focus_values, dtype=float).ravel())

  if nrows % 2 == 1:
    pos = _build_adaptive_axis(0.0, limit, nrows // 2 + 1, abs_focus, focus_strength)
    axis = np.concatenate((-pos[1:][::-1], pos))
  else:
    pos = _build_adaptive_axis(0.0, limit, nrows // 2 + 1, abs_focus, focus_strength)
    pos_nozero = pos[1:]
    axis = np.concatenate((-pos_nozero[::-1], pos_nozero))

  if axis.size > 1:
    src_min = float(axis[0])
    src_max = float(axis[-1])
    if src_max > src_min:
      axis = imag_min + (axis - src_min) * ((imag_max - imag_min) / (src_max - src_min))
    axis[0] = imag_min
    axis[-1] = imag_max

  return axis


def _validate_axis(axis, label):
  """Validate a custom axis as finite, 1D, and strictly increasing."""
  arr = np.asarray(axis, dtype=float).ravel()
  if arr.size == 0:
    raise ValueError(f'{label} must contain at least one value')
  if not np.all(np.isfinite(arr)):
    raise ValueError(f'{label} must contain only finite values')
  if arr.size > 1 and not np.all(np.diff(arr) > 0.0):
    raise ValueError(f'{label} must be strictly increasing')
  return arr


def _compute_sigmin_points(zz_flat, T, nprocs, progress_label='pseudospectrum points'):
  """Evaluate sigma_min at flattened complex sample points in parallel."""
  zz_flat = np.asarray(zz_flat).ravel()
  sigmin_flat = np.zeros((zz_flat.shape[0],))
  worker_count = _effective_worker_count(zz_flat.shape[0], nprocs)
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


def build_run_metadata(
  args,
  real_min,
  real_max,
  imag_min,
  imag_max,
  rows,
  cols,
  grid_strategy='uniform',
  adaptive_info=None,
  requested_bounds=None) -> Dict[str, Any]:
  """Build structured metadata describing one pseudospectrum run."""
  grid_points_effective = int(rows) * int(cols)
  effective_workers = _effective_worker_count(grid_points_effective, args.nprocs)
  if requested_bounds is None:
    requested_bounds = (real_min, real_max, imag_min, imag_max)
  req_re_min, req_re_max, req_im_min, req_im_max = requested_bounds

  if adaptive_info is None:
    adaptive_info = {'enabled': False}

  return {
    'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
    'hostname': socket.gethostname(),
    'cwd': os.getcwd(),
    'run_tag': args.run_tag,
    'case_tag': args.case_tag,
    'slurm_job_id': os.environ.get('SLURM_JOB_ID', ''),
    'grid_points': int(grid_points_effective),
    'grid_points_requested': int(args.grid_points),
    'grid_shape': {
      'rows': int(rows),
      'cols': int(cols),
    },
    'grid_strategy': str(grid_strategy),
    'nprocs_requested': int(args.nprocs),
    'nprocs_effective': int(effective_workers),
    'bounds': {
      'real_min': float(real_min),
      'real_max': float(real_max),
      'imag_min': float(imag_min),
      'imag_max': float(imag_max),
    },
    'requested_bounds': {
      'real_min': float(req_re_min),
      'real_max': float(req_re_max),
      'imag_min': float(req_im_min),
      'imag_max': float(req_im_max),
    },
    'levels': {
      'min_level': float(args.min_level),
    },
    'adaptive': adaptive_info,
  }


def write_run_metadata(output_dir, metadata):
  """Persist run metadata as JSON for reproducibility and traceability."""
  path = os.path.join(output_dir, 'run_metadata.json')
  with open(path, 'w', encoding='ascii') as f:
    json.dump(metadata, f, indent=2, sort_keys=True)
  print(f'wrote metadata: {path}', flush=True)


def compute_pseudospectrum(
  imat,
  grid_points=128,
  nprocs=10,
  real_min=None,
  real_max=None,
  imag_min=None,
  imag_max=None,
  real_axis=None,
  imag_axis=None):
  """Compute pseudospectrum over a rectangular complex grid.

  If the sampled imaginary axis is symmetric about zero, only the upper
  half-plane is evaluated and then mirrored to the lower half-plane.
  """
  T = imat
  use_custom_axes = (real_axis is not None) or (imag_axis is not None)
  if use_custom_axes:
    if real_axis is None or imag_axis is None:
      raise ValueError('real_axis and imag_axis must be provided together')
    r = _validate_axis(real_axis, 'real_axis')
    c = _validate_axis(imag_axis, 'imag_axis')
    ncols = int(r.size)
    nrows = int(c.size)
    real_min = float(r[0])
    real_max = float(r[-1])
    imag_min = float(c[0])
    imag_max = float(c[-1])
  else:
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

def choose_contour_levels(sigmin, min_level=1e-5, nlevels=5):
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

def build_reduction_mapping(jacobian_path):
  """Build the global boolean mask used to reduce the Jacobian and mass matrix."""
  jac = Matrix(jacobian_path, '/jacobian')
  nrg_block = jac.nrg // FIELD_BLOCK_COUNT
  keep_global = np.zeros(jac.nrg, dtype=bool)
  for i in KEPT_BLOCK_IDS:
    keep_global[nrg_block * i:nrg_block * (i + 1)] = True

  if np.any(jac.bcg):
    keep_global &= ~jac.bcg

  return int(jac.nr), keep_global

def write_restart_eigenvectors(eigvecs, keep_global, nr_local, out_dir):
  """Write reduced eigenvectors as OFT restart files."""
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

  block_tags = ['U_n', 'U_velx', 'U_vely', 'U_velz', 'U_T', 'U_psi', 'U_by']
  os.makedirs(out_dir, exist_ok=True)

  for i, vec in enumerate(eigvecs.T):
    global_vec = np.zeros((keep_global.shape[0],), dtype=vec.dtype)
    global_vec[keep_global] = vec
    global_vec = np.real(global_vec)
    # Separate the global state vector into seven OFT field blocks.
    blocks = np.split(global_vec, FIELD_BLOCK_COUNT)
    # Save OFT-compatible restart fields and metadata.
    with h5py.File(f'{out_dir}/xmhd2d_{i:05d}.rst', 'w') as f:
      f.create_dataset('OFT_idx_Version', data=np.array([1], dtype=np.int32))
      f.create_dataset('t', data=np.array([float(i)], dtype=np.float64))
      f.create_dataset('dt', data=np.array([1.0], dtype=np.float64))
      for j, block in enumerate(blocks):
        f.create_dataset(block_tags[j], data=block)

def parse_args() -> argparse.Namespace:
  """Parse command line arguments for pseudospectrum generation."""
  parser = argparse.ArgumentParser(description='Compute reduced pseudospectrum and save outputs.')
  parser.add_argument('--grid-points', type=int, default=128, help='Total number of grid points (minimum 128).')
  parser.add_argument('--adaptive-grid', action='store_true',
                      help='Enable two-pass adaptive grid focused near a contour level.')
  parser.add_argument('--adaptive-coarse-fraction', type=float, default=0.25,
                      help='Fraction of final grid points used in coarse adaptive pass (0 < value <= 1).')
  parser.add_argument('--adaptive-contour-level', type=float, default=None,
                      help='Contour level used to build adaptive focused grid (defaults to --min-level).')
  parser.add_argument('--adaptive-pad-distance', type=float, default=0.0,
                      help='Absolute padding distance from the adaptive contour envelope.')
  parser.add_argument('--adaptive-focus-strength', type=float, default=8.0,
                      help='Density weighting strength near contour projections for adaptive axes (>= 0).')
  parser.add_argument('--adaptive-save-coarse', action='store_true',
                      help='Save coarse-pass adaptive arrays for diagnostics.')
  parser.add_argument('--nprocs', type=int, default=128, help='Worker process count.')
  parser.add_argument('--real-min', type=float, required=True,
                      help='Sampled real-axis minimum.')
  parser.add_argument('--real-max', type=float, required=True,
                      help='Sampled real-axis maximum.')
  parser.add_argument('--imag-min', type=float, required=True,
                      help='Sampled imaginary-axis minimum.')
  parser.add_argument('--imag-max', type=float, required=True,
                      help='Sampled imaginary-axis maximum.')
  parser.add_argument('--min-level', type=float, default=1e-5, help='Minimum contour level.')
  parser.add_argument('--run-tag', type=str, default='', help='Batch-level run identifier for metadata tracking.')
  parser.add_argument('--case-tag', type=str, default='', help='Case identifier for metadata tracking.')
  parser.add_argument('--output-dir', type=str, default='pseudospectrum', help='Output directory for arrays and metadata.')
  return parser.parse_args()


def validate_and_normalize_args(args) -> Tuple[float, float, float, float]:
  """Validate CLI inputs and normalize output naming."""
  if args.nprocs < 1:
    raise ValueError('nprocs must be >= 1')
  if args.grid_points < 1:
    raise ValueError('grid-points must be >= 1')
  if args.min_level <= 0.0:
    raise ValueError('min-level must be positive')
  if args.adaptive_coarse_fraction <= 0.0 or args.adaptive_coarse_fraction > 1.0:
    raise ValueError('adaptive-coarse-fraction must satisfy 0 < value <= 1')
  if args.adaptive_focus_strength < 0.0:
    raise ValueError('adaptive-focus-strength must be >= 0')
  if args.adaptive_pad_distance < 0.0:
    raise ValueError('adaptive-pad-distance must be >= 0')

  if args.adaptive_contour_level is None:
    args.adaptive_contour_level = float(args.min_level)
  if args.adaptive_contour_level <= 0.0:
    raise ValueError('adaptive-contour-level must be positive')

  real_min = args.real_min
  real_max = args.real_max
  imag_min = args.imag_min
  imag_max = args.imag_max
  if real_max <= real_min or imag_max <= imag_min:
    raise ValueError(
      'region to cover must be prescribed with strict bounds: '
      '--real-max > --real-min and --imag-max > --imag-min')

  if args.adaptive_grid:
    if args.adaptive_pad_distance <= 0.0:
      raise ValueError('adaptive-pad-distance must be positive when --adaptive-grid is enabled')
    print(
      'adaptive-grid enabled: final pass ignores --real/--imag bounds and uses spectrum-guided bounds',
      flush=True)
  return real_min, real_max, imag_min, imag_max


def load_or_build_real_jacobian(jacobian_path, massmat_path, keep_global, dt=DT):
  """Load cached reduced Jacobian, or construct and cache it from HDF5 matrices."""
  try:
    real_jac = np.load(REAL_JACOBIAN_CACHE)
    print(f'loaded cached reduced Jacobian: {REAL_JACOBIAN_CACHE} shape={real_jac.shape}', flush=True)
    return real_jac
  except (FileNotFoundError, OSError, ValueError):
    pass

  mmat = Matrix(massmat_path, '/massmat')
  jac = Matrix(jacobian_path, '/jacobian')

  mmat_big = sparse.block_diag([mmat.csr_rep] * FIELD_BLOCK_COUNT, format='csr')
  del mmat

  jac_gl = make_global_mat(jac.csr_rep.toarray(), nrg=jac.nrg, ncg=jac.ncg, lg=jac.lg)
  jac_gl = sparse.csr_array(jac_gl)
  jac_gl.eliminate_zeros()

  mmat_gl = make_global_mat(mmat_big.toarray(), nrg=jac.nrg, ncg=jac.ncg, lg=jac.lg)
  mmat_gl = sparse.csr_array(mmat_gl)
  mmat_gl.eliminate_zeros()

  print('Shape of jac_gl is:', jac_gl.shape, flush=True)
  print('shape of mmat_gl is:', mmat_gl.shape, flush=True)

  reduced_jac = jac_gl[keep_global][:, keep_global].tocsr()
  reduced_mmat = mmat_gl[keep_global][:, keep_global].tocsr()

  print('Shape of reduced_Jac is:', reduced_jac.shape, flush=True)
  print('shape of reduced_Mmat is:', reduced_mmat.shape, flush=True)

  solved = sparse.linalg.spsolve(reduced_jac, reduced_mmat.toarray())
  solved = np.asarray(solved)
  identity = np.eye(reduced_jac.shape[0], dtype=solved.dtype)
  real_jac = (solved - identity) / dt

  np.save(REAL_JACOBIAN_CACHE, real_jac)
  print(f'saved reduced Jacobian cache: {REAL_JACOBIAN_CACHE}', flush=True)
  return real_jac


def load_or_compute_eigenvalues(real_jac, keep_global, nr_local, output_dir):
  """Load cached eigenvalues or compute and cache spectrum/eigenvectors."""
  try:
    eigvals = np.load(EIGVAL_CACHE)
    print(f'loaded cached eigenvalues: {EIGVAL_CACHE} shape={eigvals.shape}', flush=True)
    return eigvals
  except (FileNotFoundError, OSError, ValueError):
    pass

  print('computing full eigenvalue spectrum with scipy.linalg.eig', flush=True)
  eigvals = np.linalg.eigvals(real_jac)
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


def load_or_compute_schur_factor(real_jac):
  """Load cached Schur factorization or compute and persist it."""
  try:
    schur_t = np.load(SCHUR_CACHE)
    print(f'loaded cached Schur factor: {SCHUR_CACHE} shape={schur_t.shape}', flush=True)
    return schur_t
  except (FileNotFoundError, OSError, ValueError):
    pass

  print('computing schur factorization', flush=True)
  schur_t, _ = scipy.linalg.schur(real_jac, output='complex')
  plt.figure()
  plt.scatter(schur_t.diagonal().real, schur_t.diagonal().imag, s=2, c='k')
  plt.savefig('./full_reduced_schur_eigs.png')
  plt.close()
  np.save(SCHUR_CACHE, schur_t)
  return schur_t


def save_pseudospectrum_arrays(output_dir, R, C, sigmin):
  """Save sampled grid and pseudospectrum values as NumPy arrays."""
  np.save(os.path.join(output_dir, 'pseudo_R.npy'), R)
  np.save(os.path.join(output_dir, 'pseudo_C.npy'), C)
  np.save(os.path.join(output_dir, 'pseudo_sigmin.npy'), sigmin)


def save_adaptive_coarse_arrays(output_dir, R, C, sigmin):
  """Save coarse-pass adaptive arrays for diagnostics and reproducibility."""
  np.save(os.path.join(output_dir, 'adaptive_coarse_R.npy'), R)
  np.save(os.path.join(output_dir, 'adaptive_coarse_C.npy'), C)
  np.save(os.path.join(output_dir, 'adaptive_coarse_sigmin.npy'), sigmin)


def _build_adaptive_sampling_axes(schur_t, eigvals, args, requested_bounds):
  """Construct adaptive nonuniform axes from coarse contour extraction."""
  coarse_points = _coarse_grid_points(args.grid_points, args.adaptive_coarse_fraction)
  coarse_source = 'eigenspectrum_seed'
  try:
    coarse_re_min, coarse_re_max, coarse_im_min, coarse_im_max = _coarse_seed_bounds_from_spectrum(
      eigvals,
      args.adaptive_pad_distance)
  except ValueError:
    coarse_source = 'requested_bounds_fallback'
    coarse_re_min, coarse_re_max, coarse_im_min, coarse_im_max = requested_bounds

  print(
    f'adaptive coarse pass: points={coarse_points}, contour_level={args.adaptive_contour_level:.6g}, '
    f'bounds=([{coarse_re_min:.6g}, {coarse_re_max:.6g}], [{coarse_im_min:.6g}, {coarse_im_max:.6g}])',
    flush=True)
  coarse_R, coarse_C, coarse_sigmin = compute_pseudospectrum(
    schur_t,
    grid_points=coarse_points,
    nprocs=args.nprocs,
    real_min=coarse_re_min,
    real_max=coarse_re_max,
    imag_min=coarse_im_min,
    imag_max=coarse_im_max)

  contour_vertices = _extract_contour_vertices(
    coarse_R,
    coarse_C,
    coarse_sigmin,
    args.adaptive_contour_level)

  final_source = 'requested_bounds_fallback'
  try:
    final_re_min, final_re_max, final_im_min, final_im_max, final_source = _compute_adaptive_bounds(
      contour_vertices,
      eigvals,
      args.adaptive_pad_distance)
  except ValueError:
    final_re_min, final_re_max, final_im_min, final_im_max = requested_bounds

  nrows, ncols = _grid_shape(args.grid_points)
  finite_eigs = _finite_complex_values(eigvals)
  if finite_eigs.size > 0:
    eig_focus_re = finite_eigs.real
    eig_focus_im = finite_eigs.imag
  else:
    eig_focus_re = np.array([0.5 * (final_re_min + final_re_max)], dtype=float)
    eig_focus_im = np.array([0.5 * (final_im_min + final_im_max)], dtype=float)

  if contour_vertices.size > 0:
    focus_re = contour_vertices[:, 0]
    focus_im = contour_vertices[:, 1]
  else:
    focus_re = eig_focus_re
    focus_im = eig_focus_im

  real_axis = _build_adaptive_axis(
    final_re_min,
    final_re_max,
    ncols,
    focus_re,
    args.adaptive_focus_strength)
  imag_axis = _build_adaptive_imag_axis(
    final_im_min,
    final_im_max,
    nrows,
    focus_im,
    args.adaptive_focus_strength)

  adaptive_info = {
    'enabled': True,
    'coarse_points': int(coarse_points),
    'coarse_fraction': float(args.adaptive_coarse_fraction),
    'coarse_bounds_source': coarse_source,
    'coarse_bounds': {
      'real_min': float(coarse_re_min),
      'real_max': float(coarse_re_max),
      'imag_min': float(coarse_im_min),
      'imag_max': float(coarse_im_max),
    },
    'contour_level': float(args.adaptive_contour_level),
    'contour_vertices': int(contour_vertices.shape[0]),
    'final_bounds_source': final_source,
    'final_bounds': {
      'real_min': float(final_re_min),
      'real_max': float(final_re_max),
      'imag_min': float(final_im_min),
      'imag_max': float(final_im_max),
    },
    'pad_distance': float(args.adaptive_pad_distance),
    'focus_strength': float(args.adaptive_focus_strength),
    'axis_points': {
      'rows': int(nrows),
      'cols': int(ncols),
    },
  }

  print(
    f'adaptive final pass: contour_vertices={contour_vertices.shape[0]}, '
    f'bounds_source={final_source}, '
    f'bounds=([{final_re_min:.6g}, {final_re_max:.6g}], [{final_im_min:.6g}, {final_im_max:.6g}])',
    flush=True)

  return real_axis, imag_axis, adaptive_info, coarse_R, coarse_C, coarse_sigmin


def run_pipeline(args):
  """Run end-to-end reduced pseudospectrum workflow."""
  req_re_min, req_re_max, req_im_min, req_im_max = validate_and_normalize_args(args)
  requested_bounds = (req_re_min, req_re_max, req_im_min, req_im_max)

  jacobian_path = './lin_ops.h5'
  massmat_path = './mass_mat.h5'
  nr_local, keep_global = build_reduction_mapping(jacobian_path)

  os.makedirs(args.output_dir, exist_ok=True)

  real_jac = load_or_build_real_jacobian(jacobian_path, massmat_path, keep_global, dt=DT)
  eigvals = load_or_compute_eigenvalues(real_jac, keep_global, nr_local, args.output_dir)
  schur_t = load_or_compute_schur_factor(real_jac)
  del real_jac

  print('Running pseudospectrum', flush=True)
  adaptive_info = {'enabled': False}
  grid_strategy = 'uniform'
  if args.adaptive_grid:
    grid_strategy = 'adaptive'
    real_axis, imag_axis, adaptive_info, coarse_R, coarse_C, coarse_sigmin = _build_adaptive_sampling_axes(
      schur_t,
      eigvals,
      args,
      requested_bounds)

    if args.adaptive_save_coarse:
      save_adaptive_coarse_arrays(args.output_dir, coarse_R, coarse_C, coarse_sigmin)
      adaptive_info['coarse_arrays_saved'] = True
    else:
      adaptive_info['coarse_arrays_saved'] = False

    R, C, sigmin = compute_pseudospectrum(
      schur_t,
      nprocs=args.nprocs,
      real_axis=real_axis,
      imag_axis=imag_axis)
    final_re_min = float(real_axis[0])
    final_re_max = float(real_axis[-1])
    final_im_min = float(imag_axis[0])
    final_im_max = float(imag_axis[-1])
  else:
    R, C, sigmin = compute_pseudospectrum(
      schur_t,
      grid_points=args.grid_points,
      nprocs=args.nprocs,
      real_min=req_re_min,
      real_max=req_re_max,
      imag_min=req_im_min,
      imag_max=req_im_max)
    final_re_min = req_re_min
    final_re_max = req_re_max
    final_im_min = req_im_min
    final_im_max = req_im_max

  del schur_t

  rows, cols = int(sigmin.shape[0]), int(sigmin.shape[1])
  metadata = build_run_metadata(
    args,
    final_re_min,
    final_re_max,
    final_im_min,
    final_im_max,
    rows=rows,
    cols=cols,
    grid_strategy=grid_strategy,
    adaptive_info=adaptive_info,
    requested_bounds=requested_bounds)
  write_run_metadata(args.output_dir, metadata)

  save_pseudospectrum_arrays(args.output_dir, R, C, sigmin)


def main():
  """CLI entry point."""
  run_pipeline(parse_args())


if __name__ == '__main__':
  main()