"""Compute reduced-operator pseudospectra and export interactive plots.

This script loads OFT Jacobian and mass matrices, reduces the global system,
builds the Schur form of the reduced operator, samples the pseudospectrum over
user-defined complex-plane bounds, and writes both arrays and interactive HTML
plots. It is designed for batch execution and supports multiprocessing.
"""

import os

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"

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

try:
  import plotly.graph_objects as go
except ImportError:
  go = None


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
  vals, _ = sparse.linalg.eigsh(op, k=1, which='LM', ncv=20, tol=1e-6)
  sig_min = vals[0]
  return 1 / np.sqrt(sig_min)


def _compute_sig_point(item):
  """Worker entry point for a single complex grid point."""
  idx, z = item
  if _worker_T is None or _worker_trtrs is None:
    raise RuntimeError('worker operator state is not initialized')
  return idx, _compute_sig_for_z_from_factors(z, _worker_T, _worker_trtrs)


def _worker_count(total_points, requested_nprocs):
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


def _normalize_html_name(name, label):
  """Normalize output HTML filename and preserve prior logging behavior."""
  val = str(name).strip()
  if not val:
    raise ValueError(f'{label} must not be empty')
  if not val.lower().endswith('.html'):
    fixed = f'{val}.html'
    print(f'adjusted {label} to HTML output: {fixed}', flush=True)
    return fixed
  return val

def _compute_sigmin_points(zz_flat, T, nprocs, progress_label='pseudospectrum points'):
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

def plot_pseudospectrum(
  output_dir,
  plot_name,
  R,
  C,
  sigmin,
  eigvals,
  levels):
  """Render interactive contour plot with Plotly smoothing."""
  levels = np.asarray(levels, dtype=float).ravel()
  if levels.size == 0:
    raise ValueError('levels must contain at least one contour value')

  x_axis = np.asarray(R, dtype=float)[0, :]
  y_axis = np.asarray(C, dtype=float)[:, 0]
  sig_values = np.asarray(sigmin, dtype=float)
  positive_finite = np.isfinite(sig_values) & (sig_values > 0.0)
  log_sig = np.full(sig_values.shape, np.nan, dtype=float)
  if np.any(positive_finite):
    log_sig[positive_finite] = np.log10(sig_values[positive_finite])

  finite_log = np.isfinite(log_sig)
  if not np.any(finite_log):
    tickvals = np.array([0.0], dtype=float)
    ticktext = ['nan']
    zmin = 0.0
    zmax = 0.0
  else:
    zmin = float(np.nanmin(log_sig[finite_log]))
    zmax = float(np.nanmax(log_sig[finite_log]))
    if zmax <= zmin:
      zmax = np.nextafter(zmin, np.inf)
    tickvals = np.linspace(zmin, zmax, 6)
    ticktext = [f'{10.0 ** val:.3e}' for val in tickvals]

  fig = go.Figure()
  fig.add_trace(go.Heatmap(
    x=x_axis,
    y=y_axis,
    z=log_sig,
    zmin=zmin,
    zmax=zmax,
    zsmooth='best',
    colorscale='Viridis',
    colorbar={
      'title': 'sigmin',
      'tickmode': 'array',
      'tickvals': tickvals,
      'ticktext': ticktext,
      'ticks': 'outside',
    },
    hovertemplate='Re[z]=%{x:.6g}<br>Im[z]=%{y:.6g}<br>log10(sigmin)=%{z:.4f}<extra></extra>'))

  annotations = []
  finite_sig = np.isfinite(sig_values)
  safe_sig = np.array(sig_values, copy=True)
  safe_sig[~finite_sig] = np.nan
  has_finite_sig = bool(np.any(finite_sig))
  if has_finite_sig:
    data_min = float(np.nanmin(sig_values[finite_sig]))
    data_max = float(np.nanmax(sig_values[finite_sig]))

  for i, level in enumerate(levels):
    if (not has_finite_sig) or (level < data_min) or (level > data_max):
      continue

    fig_tmp = plt.figure()
    try:
      ax_tmp = fig_tmp.add_subplot(111)
      with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        contour_set = ax_tmp.contour(x_axis, y_axis, safe_sig, levels=[float(level)])
      segments = []
      if contour_set.allsegs:
        for seg in contour_set.allsegs[0]:
          if seg is None or seg.shape[0] < 2:
            continue
          seg_arr = np.asarray(seg[:, :2], dtype=float)
          keep = np.isfinite(seg_arr).all(axis=1)
          seg_keep = seg_arr[keep]
          if seg_keep.shape[0] >= 2:
            segments.append(seg_keep)
    finally:
      plt.close(fig_tmp)

    if not segments:
      continue

    for j, seg in enumerate(segments):
      fig.add_trace(go.Scatter(
        x=seg[:, 0],
        y=seg[:, 1],
        mode='lines',
        hoverinfo='skip',
        showlegend=(i == 0 and j == 0),
        name='contours',
        line={'color': 'black', 'width': 1, 'shape': 'spline', 'smoothing': 1.0}))

    seg_lengths = []
    for seg in segments:
      delta = np.diff(seg, axis=0)
      seg_lengths.append(float(np.sum(np.hypot(delta[:, 0], delta[:, 1]))))
    longest = segments[int(np.argmax(seg_lengths))]

    delta = np.diff(longest, axis=0)
    steps = np.hypot(delta[:, 0], delta[:, 1])
    total = float(np.sum(steps))
    if total <= 0.0:
      pt = np.array(longest[longest.shape[0] // 2], dtype=float)
    else:
      cum = np.concatenate(([0.0], np.cumsum(steps)))
      target = 0.5 * total
      idx = int(np.searchsorted(cum, target, side='right') - 1)
      idx = min(max(idx, 0), steps.size - 1)
      span = steps[idx]
      if span <= 0.0:
        pt = np.array(longest[idx], dtype=float)
      else:
        frac = (target - cum[idx]) / span
        pt = np.array(longest[idx] + frac * (longest[idx + 1] - longest[idx]), dtype=float)

    if np.all(np.isfinite(pt)):
      annotations.append({
        'x': float(pt[0]),
        'y': float(pt[1]),
        'text': f'{float(level):.2e}',
        'showarrow': False,
        'xanchor': 'center',
        'yanchor': 'middle',
        'font': {'size': 10, 'color': 'black'},
        'bgcolor': 'rgba(255,255,255,0.92)',
        'borderwidth': 0,
      })

  eigvals_arr = np.asarray(eigvals).ravel()
  if eigvals_arr.size > 0:
    mask = np.isfinite(eigvals_arr.real) & np.isfinite(eigvals_arr.imag)
    eigvals_arr = eigvals_arr[mask]
  if eigvals_arr.size > 0:
    fig.add_trace(go.Scatter(
      x=eigvals_arr.real,
      y=eigvals_arr.imag,
      mode='markers',
      name='eigenvalues',
      marker={'size': 2.5, 'color': 'black', 'opacity': 0.65},
      hovertemplate='Re[lambda]=%{x:.6g}<br>Im[lambda]=%{y:.6g}<extra></extra>'))

  fig.update_layout(
    title='Pseudospectrum contour (interactive)',
    xaxis={'title': 'Re[z]', 'tickformat': '.3g', 'constrain': 'domain'},
    yaxis={'title': 'Im[z]', 'tickformat': '.3g'},
    margin={'l': 60, 'r': 200, 'b': 55, 't': 50},
    legend={'x': 0.01, 'y': 0.99},
    dragmode='zoom',
    annotations=annotations)

  out_path = os.path.join(output_dir, plot_name)
  fig.write_html(out_path, include_plotlyjs='cdn')
  print(f'wrote interactive contour: {out_path}', flush=True)

def build_metadata(
  args,
  real_min,
  real_max,
  imag_min,
  imag_max,
  rows,
  cols) -> Dict[str, Any]:
  """Build structured metadata describing one pseudospectrum run."""
  grid_points_effective = int(rows) * int(cols)
  effective_workers = _worker_count(grid_points_effective, args.nprocs)

  metadata = {
    'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
    'hostname': socket.gethostname(),
    'cwd': os.getcwd(),
    'run_tag': args.run_tag,
    'case_tag': args.case_tag,
    'slurm_job_id': os.environ.get('SLURM_JOB_ID', ''),
    'grid_points': int(grid_points_effective),
    'grid_shape': {
      'rows': int(rows),
      'cols': int(cols),
    },
    'nprocs_requested': int(args.nprocs),
    'nprocs_effective': int(effective_workers),
    'bounds': {
      'real_min': float(real_min),
      'real_max': float(real_max),
      'imag_min': float(imag_min),
      'imag_max': float(imag_max),
    },
    'levels': {
      'min_level': float(args.min_level),
      'nlevels': int(args.nlevels),
    },
  }

  return metadata


def write_metadata(output_dir, metadata):
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
  imag_max=None):
  """Compute pseudospectrum over a rectangular complex grid.

  If the sampled imaginary axis is symmetric about zero, only the upper
  half-plane is evaluated and then mirrored to the lower half-plane.
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
  parser.add_argument('--nprocs', type=int, default=128, help='Worker process count.')
  parser.add_argument('--real-min', type=float, required=True,
                      help='Sampled real-axis minimum.')
  parser.add_argument('--real-max', type=float, required=True,
                      help='Sampled real-axis maximum.')
  parser.add_argument('--imag-min', type=float, required=True,
                      help='Sampled imaginary-axis minimum.')
  parser.add_argument('--imag-max', type=float, required=True,
                      help='Sampled imaginary-axis maximum.')
  parser.add_argument('--nlevels', type=int, default=8,
                      help='Number of contour levels for interactive pseudospectrum plot.')
  parser.add_argument('--min-level', type=float, default=1e-5, help='Minimum contour level.')
  parser.add_argument('--plot-name', type=str, default='pseudoplot_default.html',
                      help='Interactive contour plot output filename.')
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
  if args.nlevels < 1:
    raise ValueError('nlevels must be >= 1')
  if args.min_level <= 0.0:
    raise ValueError('min-level must be positive')

  real_min = args.real_min
  real_max = args.real_max
  imag_min = args.imag_min
  imag_max = args.imag_max
  if real_max <= real_min or imag_max <= imag_min:
    raise ValueError(
      'region to cover must be prescribed with strict bounds: '
      '--real-max > --real-min and --imag-max > --imag-min')

  args.plot_name = _normalize_html_name(args.plot_name, 'plot-name')
  return real_min, real_max, imag_min, imag_max


def load_or_build_jacobian(jacobian_path, massmat_path, keep_global):
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
  real_jac = (solved - identity) / DT

  np.save(REAL_JACOBIAN_CACHE, real_jac)
  print(f'saved reduced Jacobian cache: {REAL_JACOBIAN_CACHE}', flush=True)
  return real_jac


def load_or_compute_eigvals(real_jac, keep_global, nr_local, output_dir):
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


def load_or_compute_schur(real_jac):
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


def run_pipeline(args):
  """Run end-to-end reduced pseudospectrum workflow."""
  real_min, real_max, imag_min, imag_max = validate_and_normalize_args(args)

  jacobian_path = './lin_ops.h5'
  massmat_path = './mass_mat.h5'
  nr_local, keep_global = build_reduction_mapping(jacobian_path)

  os.makedirs(args.output_dir, exist_ok=True)

  real_jac = load_or_build_jacobian(jacobian_path, massmat_path, keep_global)
  eigvals = load_or_compute_eigvals(real_jac, keep_global, nr_local, args.output_dir)
  schur_t = load_or_compute_schur(real_jac)
  del real_jac

  print('Running pseudospectrum', flush=True)
  R, C, sigmin = compute_pseudospectrum(
    schur_t,
    grid_points=args.grid_points,
    nprocs=args.nprocs,
    real_min=real_min,
    real_max=real_max,
    imag_min=imag_min,
    imag_max=imag_max)

  del schur_t

  rows, cols = int(sigmin.shape[0]), int(sigmin.shape[1])
  metadata = build_metadata(
    args,
    real_min,
    real_max,
    imag_min,
    imag_max,
    rows=rows,
    cols=cols)

  if go is None:
    raise ImportError('plotly is required for interactive plots; install plotly')

  levels = choose_contour_levels(sigmin, min_level=args.min_level, nlevels=args.nlevels)
  print(
    f'plot_pseudospectrum: levels={np.array2string(levels, precision=3)}, '
    f'xlim=({real_min:.6g}, {real_max:.6g}), '
    f'ylim=({imag_min:.6g}, {imag_max:.6g})',
    flush=True)

  plot_pseudospectrum(
    args.output_dir,
    args.plot_name,
    R,
    C,
    sigmin,
    eigvals,
    levels)

  metadata['levels']['values'] = [float(v) for v in np.asarray(levels, dtype=float).ravel()]
  plot_info = {
    'enabled': True,
    'plot_name': args.plot_name,
  }
  metadata['plot'] = plot_info

  write_metadata(args.output_dir, metadata)
  save_pseudospectrum_arrays(args.output_dir, R, C, sigmin)


def main():
  """CLI entry point."""
  run_pipeline(parse_args())


if __name__ == '__main__':
  main()