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

def _split_plot_output_names(plot_name: str) -> Tuple[str, str]:
  """Derive separate output names for heatmap and contour views."""
  stem, ext = os.path.splitext(str(plot_name))
  ext = ext if ext else '.html'
  return f'{stem}_heatmap{ext}', f'{stem}_contours{ext}'


def _add_eigenvalue_overlay(fig, eigvals):
  """Overlay eigenvalue markers on an existing Plotly figure."""
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


def pseudo_heatmap(output_dir, plot_name, R, C, sigmin, eigvals):
  """Render an interactive heatmap view of pseudospectrum values."""
  title_font_size = 20
  axis_title_font_size = 18
  axis_tick_font_size = 14
  colorbar_title_font_size = 18
  colorbar_tick_font_size = 14

  x_axis = np.asarray(R, dtype=float)[0, :]
  y_axis = np.asarray(C, dtype=float)[:, 0]
  x_min = float(np.min(x_axis))
  x_max = float(np.max(x_axis))
  y_min = float(np.min(y_axis))
  y_max = float(np.max(y_axis))

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

  fig = go.Figure(go.Heatmap(
    x=x_axis,
    y=y_axis,
    z=log_sig,
    zmin=zmin,
    zmax=zmax,
    zsmooth='best',
    colorscale='Viridis',
    colorbar={
      'title': {
        'text': 'ε',
        'font': {'size': colorbar_title_font_size},
      },
      'tickmode': 'array',
      'tickvals': tickvals,
      'ticktext': ticktext,
      'tickfont': {'size': colorbar_tick_font_size},
      'ticks': '',
      'ticklen': 0,
    },
    hovertemplate='Re[z]=%{x:.6g}<br>Im[z]=%{y:.6g}<br>log10(epsilon)=%{z:.4f}<extra></extra>'))

  _add_eigenvalue_overlay(fig, eigvals)

  fig.update_layout(
    title={
      'text': 'Pseudospectra Heatmap of Resistive MHD Operator',
      'x': 0.5,
      'xanchor': 'center',
      'font': {'size': title_font_size},
    },
    xaxis={
      'title': {'text': 'Re[z]', 'font': {'size': axis_title_font_size}},
      'tickfont': {'size': axis_tick_font_size},
      'tickformat': '.3g',
      'constrain': 'domain',
      'autorange': False,
      'range': [x_min, x_max],
    },
    yaxis={
      'title': {'text': 'Im[z]', 'font': {'size': axis_title_font_size}},
      'tickfont': {'size': axis_tick_font_size},
      'tickformat': '.3g',
      'autorange': False,
      'range': [y_min, y_max],
    },
    margin={'l': 60, 'r': 200, 'b': 55, 't': 50},
    dragmode='zoom')

  out_path = os.path.join(output_dir, plot_name)
  fig.write_html(out_path, include_plotlyjs='cdn')
  print(f'wrote interactive heatmap: {out_path}', flush=True)


def pseudo_contours(output_dir, plot_name, R, C, sigmin, eigvals, levels):
  """Render an interactive, color-coded contour view of pseudospectrum values."""
  levels = np.asarray(levels, dtype=float).ravel()
  if levels.size == 0:
    raise ValueError('levels must contain at least one contour value')
  levels = np.unique(np.sort(levels))
  if levels[0] <= 0.0:
    raise ValueError('levels must be strictly positive')

  # Keep log-space color progression for wide dynamic range while directly
  # labeling contour lines in epsilon units.
  log_levels = np.log10(levels)
  contour_start = float(log_levels[0])
  contour_end = float(log_levels[-1])

  title_font_size = 20
  axis_title_font_size = 18
  axis_tick_font_size = 14
  colorbar_title_font_size = 18
  colorbar_tick_font_size = 14
  contour_label_font_size = 12

  x_axis = np.asarray(R, dtype=float)[0, :]
  y_axis = np.asarray(C, dtype=float)[:, 0]
  x_min = float(np.min(x_axis))
  x_max = float(np.max(x_axis))
  y_min = float(np.min(y_axis))
  y_max = float(np.max(y_axis))

  sig_values = np.asarray(sigmin, dtype=float)
  finite_positive = np.isfinite(sig_values) & (sig_values > 0.0)
  safe_sig = np.array(sig_values, copy=True)
  safe_sig[~finite_positive] = np.nan
  safe_log_sig = np.array(sig_values, copy=True)
  safe_log_sig[finite_positive] = np.log10(sig_values[finite_positive])
  safe_log_sig[~finite_positive] = np.nan

  level_tick_vals = np.unique(np.concatenate(([contour_start, contour_end], log_levels)))
  level_tick_text = [f'{10.0 ** val:.1e}' for val in level_tick_vals]

  if levels.size == 1:
    contour_end = float(np.nextafter(contour_start, np.inf))
    contour_size = float(contour_end - contour_start)
  else:
    contour_size = float((contour_end - contour_start) / (levels.size - 1))
  if contour_size <= 0.0:
    contour_size = float(np.nextafter(0.0, 1.0))

  fig = go.Figure()
  fig.add_trace(go.Contour(
    x=x_axis,
    y=y_axis,
    z=safe_log_sig,
    customdata=safe_sig,
    zmin=contour_start,
    zmax=contour_end,
    autocontour=False,
    colorscale='Turbo',
    contours={
      'start': contour_start,
      'end': contour_end,
      'size': contour_size,
      'showlabels': False,
      'coloring': 'lines',
    },
    line={'width': 2.0},
    colorbar={
      'title': {
        'text': 'ε',
        'font': {'size': colorbar_title_font_size},
      },
      'tickmode': 'array',
      'tickvals': level_tick_vals,
      'ticktext': level_tick_text,
      'tickfont': {'size': colorbar_tick_font_size},
      'ticks': '',
      'ticklen': 0,
    },
    hovertemplate=(
      'Re[z]=%{x:.6g}<br>Im[z]=%{y:.6g}<br>'
      'epsilon=%{customdata:.4e}<br>log10(epsilon)=%{z:.4f}<extra></extra>')))

  # Overlay one contour trace per level to label lines directly in epsilon units.
  for level in levels:
    level = float(level)
    label_end = float(np.nextafter(level, np.inf))
    label_size = float(max(label_end - level, np.finfo(float).eps))
    fig.add_trace(go.Contour(
      x=x_axis,
      y=y_axis,
      z=safe_sig,
      autocontour=False,
      showscale=False,
      hoverinfo='skip',
      contours={
        'start': level,
        'end': label_end,
        'size': label_size,
        'showlabels': True,
        'labelformat': '.2e',
        'labelfont': {'size': contour_label_font_size, 'color': 'black'},
        'coloring': 'none',
      },
      line={'width': 0.0, 'color': 'rgba(0,0,0,0)'}))

  _add_eigenvalue_overlay(fig, eigvals)

  fig.update_layout(
    title={
      'text': 'Pseudospectra Contours of Resistive MHD Operator',
      'x': 0.5,
      'xanchor': 'center',
      'font': {'size': title_font_size},
    },
    xaxis={
      'title': {'text': 'Re[z]', 'font': {'size': axis_title_font_size}},
      'tickfont': {'size': axis_tick_font_size},
      'tickformat': '.3g',
      'constrain': 'domain',
      'autorange': False,
      'range': [x_min, x_max],
    },
    yaxis={
      'title': {'text': 'Im[z]', 'font': {'size': axis_title_font_size}},
      'tickfont': {'size': axis_tick_font_size},
      'tickformat': '.3g',
      'autorange': False,
      'range': [y_min, y_max],
    },
    margin={'l': 60, 'r': 200, 'b': 55, 't': 50},
    dragmode='zoom')

  out_path = os.path.join(output_dir, plot_name)
  fig.write_html(out_path, include_plotlyjs='cdn')
  print(f'wrote interactive contours: {out_path}', flush=True)


def build_metadata(
  args,
  real_min,
  real_max,
  imag_min,
  imag_max,
  grid_points,
  rows=None,
  cols=None,
  grid_type='structured',
  grid_source='generated') -> Dict[str, Any]:
  """Build structured metadata describing one pseudospectrum run."""
  grid_points_effective = int(grid_points)
  effective_workers = _worker_count(grid_points_effective, args.nprocs)
  grid_shape = None
  if rows is not None and cols is not None:
    grid_shape = {
      'rows': int(rows),
      'cols': int(cols),
    }

  metadata = {
    'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
    'hostname': socket.gethostname(),
    'cwd': os.getcwd(),
    'run_tag': args.run_tag,
    'case_tag': args.case_tag,
    'slurm_job_id': os.environ.get('SLURM_JOB_ID', ''),
    'grid_points': int(grid_points_effective),
    'grid_shape': grid_shape,
    'grid_type': grid_type,
    'grid_source': grid_source,
    'grid_npy': args.grid_npy or '',
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

def choose_contour_levels(sigmin, min_level=1e-7, nlevels=5):
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
  parser.add_argument('--grid-npy', type=str, default='',
                      help='Optional .npy file containing a flat complex grid to sample.')
  parser.add_argument('--grid-shape', type=int, nargs=2, metavar=('ROWS', 'COLS'),
                      help='Optional reshape for structured plots when using --grid-npy.')
  parser.add_argument('--real-min', type=float, default=None,
                      help='Sampled real-axis minimum (required unless --grid-npy is used).')
  parser.add_argument('--real-max', type=float, default=None,
                      help='Sampled real-axis maximum (required unless --grid-npy is used).')
  parser.add_argument('--imag-min', type=float, default=None,
                      help='Sampled imaginary-axis minimum (required unless --grid-npy is used).')
  parser.add_argument('--imag-max', type=float, default=None,
                      help='Sampled imaginary-axis maximum (required unless --grid-npy is used).')
  parser.add_argument('--nlevels', type=int, default=16,
                      help='Number of contour levels for interactive pseudospectrum plot.')
  parser.add_argument('--min-level', type=float, default=1e-7, help='Minimum contour level.')
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

  if args.grid_npy:
    if args.grid_shape is not None:
      rows, cols = args.grid_shape
      if rows < 1 or cols < 1:
        raise ValueError('grid-shape rows and cols must be >= 1')
    real_min = None
    real_max = None
    imag_min = None
    imag_max = None
  else:
    real_min = args.real_min
    real_max = args.real_max
    imag_min = args.imag_min
    imag_max = args.imag_max
    if real_min is None or real_max is None or imag_min is None or imag_max is None:
      raise ValueError('real/imag bounds must be provided unless --grid-npy is used')
    if real_max <= real_min or imag_max <= imag_min:
      raise ValueError(
        'region to cover must be prescribed with strict bounds: '
        '--real-max > --real-min and --imag-max > --imag-min')

  args.plot_name = _normalize_html_name(args.plot_name, 'plot-name')
  return real_min, real_max, imag_min, imag_max


def load_flat_grid_npy(path: str) -> np.ndarray:
  """Load a flat complex grid from a .npy file."""
  zz = np.load(path, allow_pickle=False)
  if zz.size == 0:
    raise ValueError('grid-npy must contain at least one complex value')
  zz = np.asarray(zz).ravel()
  if not np.iscomplexobj(zz):
    zz = zz.astype(np.complex128)
  return zz


def grid_bounds_from_flat(zz_flat: np.ndarray) -> Tuple[float, float, float, float]:
  """Compute real/imag bounds from a flat complex grid."""
  real_min = float(np.min(zz_flat.real))
  real_max = float(np.max(zz_flat.real))
  imag_min = float(np.min(zz_flat.imag))
  imag_max = float(np.max(zz_flat.imag))
  return real_min, real_max, imag_min, imag_max


def save_pseudospectrum_flat(output_dir, zz_flat, sigmin_flat):
  """Save flat complex grid and pseudospectrum values as NumPy arrays."""
  np.save(os.path.join(output_dir, 'pseudo_z.npy'), zz_flat)
  np.save(os.path.join(output_dir, 'pseudo_sigmin_flat.npy'), sigmin_flat)


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
  plot_enabled = True
  if args.grid_npy:
    zz_flat = load_flat_grid_npy(args.grid_npy)
    real_min, real_max, imag_min, imag_max = grid_bounds_from_flat(zz_flat)
    sigmin_flat = _compute_sigmin_points(
      zz_flat,
      schur_t,
      nprocs=args.nprocs,
      progress_label='pseudospectrum points')
    if args.grid_shape is None:
      plot_enabled = False
      R = None
      C = None
      sigmin = sigmin_flat
      rows = None
      cols = None
    else:
      rows, cols = args.grid_shape
      if rows * cols != zz_flat.size:
        raise ValueError('grid-shape does not match the size of grid-npy')
      z_grid = zz_flat.reshape(rows, cols)
      R = z_grid.real
      C = z_grid.imag
      sigmin = sigmin_flat.reshape(rows, cols)
    grid_points = int(zz_flat.size)
    grid_type = 'structured' if plot_enabled else 'unstructured'
    grid_source = 'npy'
  else:
    R, C, sigmin = compute_pseudospectrum(
      schur_t,
      grid_points=args.grid_points,
      nprocs=args.nprocs,
      real_min=real_min,
      real_max=real_max,
      imag_min=imag_min,
      imag_max=imag_max)
    rows, cols = int(sigmin.shape[0]), int(sigmin.shape[1])
    grid_points = int(rows * cols)
    grid_type = 'structured'
    grid_source = 'generated'

  del schur_t

  metadata = build_metadata(
    args,
    real_min,
    real_max,
    imag_min,
    imag_max,
    grid_points=grid_points,
    rows=rows,
    cols=cols,
    grid_type=grid_type,
    grid_source=grid_source)

  if plot_enabled:
    if go is None:
      raise ImportError('plotly is required for interactive plots; install plotly')

    levels = choose_contour_levels(sigmin, min_level=args.min_level, nlevels=args.nlevels)
    print(
      f'plotting levels={np.array2string(levels, precision=3)}, '
      f'xlim=({real_min:.6g}, {real_max:.6g}), '
      f'ylim=({imag_min:.6g}, {imag_max:.6g})',
      flush=True)

    heatmap_plot_name, contour_plot_name = _split_plot_output_names(args.plot_name)
    pseudo_heatmap(
      args.output_dir,
      heatmap_plot_name,
      R,
      C,
      sigmin,
      eigvals)
    pseudo_contours(
      args.output_dir,
      contour_plot_name,
      R,
      C,
      sigmin,
      eigvals,
      levels)

    metadata['levels']['values'] = [float(v) for v in np.asarray(levels, dtype=float).ravel()]
    plot_info = {
      'enabled': True,
      'plot_name': args.plot_name,
      'plot_name_heatmap': heatmap_plot_name,
      'plot_name_contours': contour_plot_name,
    }
    metadata['plot'] = plot_info
  else:
    metadata['plot'] = {
      'enabled': False,
      'reason': 'unstructured grid; provide --grid-shape to enable plots',
    }

  write_metadata(args.output_dir, metadata)
  if plot_enabled:
    save_pseudospectrum_arrays(args.output_dir, R, C, sigmin)
  else:
    save_pseudospectrum_flat(args.output_dir, zz_flat, sigmin)


def main():
  """CLI entry point."""
  run_pipeline(parse_args())


if __name__ == '__main__':
  main()