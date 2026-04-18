import os
os.environ["OMP_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse
import json
from datetime import datetime, timezone
import socket
import h5py
import scipy
import scipy.sparse as sparse
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import multiprocessing
''' Computes the pseudospectrum of a reduced linear operator obtained from a Jacobian matrix, and saves the results as interactive HTML plots.
The script performs the following steps:
1. Loads the Jacobian matrix from an HDF5 file and constructs a global matrix representation.
2. Builds a reduction mapping to identify which global degrees of freedom to keep based on the structure of the Jacobian.
3. Computes the pseudospectrum by evaluating the smallest singular value of zI - T across a grid of complex points, where T is the reduced operator.
4. Chooses contour levels for plotting based on the computed pseudospectrum values.
5. Generates an interactive contour/heatmap pseudospectrum plot using Plotly and saves it as HTML.
The script is designed to be run in a batch environment, and it uses multiprocessing to parallelize the computation of the pseudospectrum across multiple CPU cores. The results include metadata about the run, which is saved in a JSON file for record-keeping and reproducibility.
Usage: python lin_ops_full_reduction.py --jacobian_path path/to/jacobian.h5 --output_dir path/to/output --run_tag my_run --case_tag my_case
'''
class Matrix:
  def __init__(self, filename, mat_name):
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
  def spy(self):
    plt.spy(self.csr_rep, markersize = 1, precision='present')
    plt.savefig(f'{self.mat_name}_spy.png')

# assemble global matrix from local matrix using the lg mapping
@njit
def make_global_mat(inmat, nrg=None, ncg=None, lg=None):
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
  global _worker_T, _worker_trtrs
  if _worker_T is None:
    raise RuntimeError('worker did not inherit operator matrix')
  _worker_trtrs, = scipy.linalg.get_lapack_funcs(('trtrs',), (_worker_T,))

# Compute the smallest singular value of zI - T using the Schur factorization and the trtrs LAPACK routine for triangular solves.
# This is done by defining a linear operator that applies the inverse of zI - T to a vector, and then using sparse.linalg.eigsh
# to compute the largest eigenvalue of this operator, which corresponds to the smallest singular value of zI - T.
def _compute_sig_for_z_from_factors(z, T, trtrs):
  # Avoid materializing z*I, which creates an extra dense allocation.
  T1 = -T.copy()
  T1.flat[::T1.shape[0] + 1] += z

  def _matvec(q):
    tmp, _ = trtrs(T1, q.reshape(-1, 1), lower=0, trans=2, unitdiag=0)
    result, _ = trtrs(T1, tmp, lower=0, trans=0, unitdiag=0)
    return result.ravel()
  
  op = scipy.sparse.linalg.LinearOperator(
    T1.shape,
    matvec=_matvec,
    dtype=np.complex128)
  vals, _ = sparse.linalg.eigsh(op, k=1, which='LM', ncv=10, tol=1e-8)
  sig_min = vals[0]
  return 1/np.sqrt(sig_min)

def _compute_sig_for_z(z):
  return _compute_sig_for_z_from_factors(z, _worker_T, _worker_trtrs)

def _compute_sig_point(item):
  idx, z = item
  return idx, _compute_sig_for_z(z)

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

def build_run_metadata(args, real_min, real_max, imag_min, imag_max):
  effective_workers = _effective_worker_count(args.grid_points, args.nprocs)
  rows, cols = _grid_shape(args.grid_points)
  return {
    'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
    'hostname': socket.gethostname(),
    'cwd': os.getcwd(),
    'run_tag': args.run_tag,
    'case_tag': args.case_tag,
    'slurm_job_id': os.environ.get('SLURM_JOB_ID', ''),
    'grid_points': int(args.grid_points),
    'grid_points_requested': int(args.grid_points),
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
    'plot': {
      'plot_name': args.plot_name,
    },
  }

def write_run_metadata(output_dir, metadata):
  path = os.path.join(output_dir, 'run_metadata.json')
  with open(path, 'w', encoding='ascii') as f:
    json.dump(metadata, f, indent=2, sort_keys=True)
  print(f'wrote metadata: {path}', flush=True)

# Compute the pseudospectrum by iterating over a grid of points 
# in the complex plane, and for each point, 
# computing the smallest singular value of zI - T using the 
# _compute_sig_for_z function.
def compute_pseudospectrum(
  imat,
  grid_points=128,
  nprocs=10,
  real_min=None,
  real_max=None,
  imag_min=None,
  imag_max=None):
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
  R, C =  np.meshgrid(r, c)
  zz = R + 1j * C
  zz_flat = zz.ravel()
  sigmin_flat = np.zeros((zz_flat.shape[0],))
  worker_count = _effective_worker_count(zz_flat.shape[0], nprocs)
  print(
    f"computing pseudospectrum on Re[z] in [{real_min:.6g}, {real_max:.6g}] "
    f"and Im[z] in [{imag_min:.6g}, {imag_max:.6g}] with shape=({nrows},{ncols})",
    flush=True)

  print(
    f"using nprocs={worker_count}, work_items={zz_flat.shape[0]}",
    flush=True)

  report_every = max(1, zz_flat.shape[0] // 10)
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
        if completed % report_every == 0 or completed == zz_flat.shape[0]:
          print(f'completed {completed}/{zz_flat.shape[0]} pseudospectrum points', flush=True)
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

  sigmin = (sigmin_flat.reshape(nrows, ncols) - 1) / dt
  return R, C, sigmin

def choose_contour_levels(sigmin, min_level=1e-5, nlevels=5):
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

def plot_pseudospectrum(
  R,
  C,
  sigmin,
  output_path,
  eigvals,
  auto_open=False,
  eig_marker_size=2.5,
  min_level=1e-12,
  nlevels=5):
  try:
    import plotly.graph_objects as go
  except ImportError as exc:
    raise ImportError('plotly is required for interactive contour export (pip install plotly)') from exc

  root, ext = os.path.splitext(output_path)
  if ext.lower() != '.html':
    output_path = root + '.html'

  levels = choose_contour_levels(sigmin, min_level=min_level, nlevels=nlevels)
  print(
    f'plot_pseudospectrum: levels={np.array2string(levels, precision=3)}, '
    f'xlim=({float(np.min(R)):.6g}, {float(np.max(R)):.6g}), '
    f'ylim=({float(np.min(C)):.6g}, {float(np.max(C)):.6g})',
    flush=True)

  sig = np.asarray(sigmin, dtype=float)
  mask = np.isfinite(sig) & (sig > 0)
  if not np.any(mask):
    raise ValueError('sigmin has no finite positive entries for interactive contour plotting')

  x_axis = np.asarray(R, dtype=float)
  y_axis = np.asarray(C, dtype=float)
  if x_axis.ndim == 2:
    x_axis = x_axis[0, :]
  if y_axis.ndim == 2:
    y_axis = y_axis[:, 0]

  log_sig = np.full(sig.shape, np.nan, dtype=float)
  log_sig[mask] = np.log10(sig[mask])
  finite_log = log_sig[mask]

  color_lo = float(np.min(finite_log))
  color_hi = float(np.max(finite_log))
  if color_hi <= color_lo:
    color_hi = np.nextafter(color_lo, np.inf)

  color_ticks = np.linspace(color_lo, color_hi, num=6)
  color_tick_pairs = []
  for tv in color_ticks:
    if color_tick_pairs and np.isclose(float(tv), color_tick_pairs[-1][0], rtol=0.0, atol=1e-12):
      continue
    color_tick_pairs.append((float(tv), f'{(10.0 ** float(tv)):.3e}'))
  if not color_tick_pairs:
    color_tick_pairs = [(color_lo, f'{float(np.min(sig[mask])):.3e}')]

  fig = go.Figure()
  fig.add_trace(go.Heatmap(
    x=x_axis,
    y=y_axis,
    z=log_sig,
    customdata=sig,
    zsmooth='best',
    connectgaps=True,
    zmin=color_lo,
    zmax=color_hi,
    colorscale='Viridis',
    colorbar=dict(
      title='sigmin',
      tickmode='array',
      tickvals=[p[0] for p in color_tick_pairs],
      ticktext=[p[1] for p in color_tick_pairs],
      ticks='outside'),
    hovertemplate='Re[z]=%{x:.6g}<br>Im[z]=%{y:.6g}<br>sigmin=%{customdata:.3e}<extra></extra>'))

  for lv in np.asarray(levels, dtype=float):
    fig.add_trace(go.Contour(
      x=x_axis,
      y=y_axis,
      z=sig,
      autocontour=False,
      contours=dict(
        start=float(lv),
        end=float(lv),
        size=1.0,
        coloring='none',
        showlines=True,
        showlabels=True,
        labelformat='.2e'),
      line=dict(width=1, color='black'),
      showscale=False,
      showlegend=False,
      name='contour',
      hovertemplate=f'contour level={float(lv):.3e}<extra></extra>'))

  vals = np.asarray(eigvals)
  vals = vals[np.isfinite(vals)]
  if vals.size > 0:
    fig.add_trace(go.Scatter(
      x=vals.real,
      y=vals.imag,
      mode='markers',
      marker=dict(size=float(eig_marker_size), color='black', opacity=0.65),
      name='eigenvalues',
      hovertemplate='Re[lambda]=%{x:.6g}<br>Im[lambda]=%{y:.6g}<extra></extra>'))

  fig.update_layout(
    title='Pseudospectrum contour (interactive)',
    xaxis=dict(title='Re[z]', tickformat='.3g', constrain='domain'),
    yaxis=dict(title='Im[z]', tickformat='.3g'),
    template='plotly_white',
    dragmode='zoom',
    margin=dict(l=60, r=20, b=55, t=50),
    legend=dict(x=0.01, y=0.99))

  fig.write_html(
    output_path,
    include_plotlyjs='cdn',
    full_html=True,
    auto_open=auto_open)
  print(f'wrote interactive contour: {output_path}', flush=True)

def build_reduction_mapping(jacobian_path):
  jac = Matrix(jacobian_path, '/jacobian')
  nrg_block = jac.nrg // 7
  keep_global = np.zeros(jac.nrg, dtype=bool)
  for i in (0, 1, 3, 4, 5):
    keep_global[nrg_block * i:nrg_block * (i + 1)] = True

  if np.any(jac.bcg):
    keep_global &= ~jac.bcg

  return int(jac.nr), keep_global

def get_plot_vecs(eigvecs, keep_global, out_dir):
  # Convert reduced eigenvectors to global, then back to local ordering.
  nred = int(np.count_nonzero(keep_global))
  if eigvecs.shape[0] != nred:
    raise ValueError(
      f'eigvec rows ({eigvecs.shape[0]}) do not match reduced size ({nred})')
  if nr_local % 7 != 0:
    raise ValueError(f'nr_local ({nr_local}) must be divisible by 7')

  block_tags = ['U_n', 'U_velx', 'U_vely', 'U_velz', 'U_T', 'U_psi', 'U_by']

  for i, vec in enumerate(eigvecs.T):
    global_vec = np.zeros((keep_global.shape[0],), dtype=vec.dtype)
    global_vec[keep_global] = vec
    global_vec = np.real(global_vec)
    # separate the vector into seven blocks
    blocks = np.split(global_vec, 7)
    # Save OFT-compatible restart fields and metadata.
    with h5py.File(f'{out_dir}/xmhd2d_{i:05d}.rst', 'w') as f:
      f.create_dataset('OFT_idx_Version', data=np.array([1], dtype=np.int32))
      f.create_dataset('t', data=np.array([float(i)], dtype=np.float64))
      f.create_dataset('dt', data=np.array([1.0], dtype=np.float64))
      for j, block in enumerate(blocks):
        f.create_dataset(block_tags[j], data=block)

if __name__=="__main__":
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
  parser.add_argument('--min-level', type=float, default=1e-5, help='Minimum contour level.')
  parser.add_argument('--nlevels', type=int, default=5, help='Number of decade-spaced contour levels.')
  parser.add_argument('--run-tag', type=str, default='', help='Batch-level run identifier for metadata tracking.')
  parser.add_argument('--case-tag', type=str, default='', help='Case identifier for metadata tracking.')
  parser.add_argument('--output-dir', type=str, default='pseudospectrum', help='Output directory for arrays and plot.')
  parser.add_argument('--plot-name', type=str, default='pseudoplot.html',
                      help='Interactive contour output filename (HTML).')
  parser.add_argument('--plot-auto-open', action='store_true',
                      help='Attempt to open the interactive contour plot in a browser after writing it.')
  parser.add_argument('--eig-marker-size', type=float, default=2.5,
                      help='Marker size for overlaid eigenvalues in the interactive contour plot.')
  args = parser.parse_args()

  if args.nprocs < 1:
    raise ValueError('nprocs must be >= 1')
  if args.eig_marker_size <= 0.0:
    raise ValueError('eig-marker-size must be positive')

  real_min = args.real_min
  real_max = args.real_max
  imag_min = args.imag_min
  imag_max = args.imag_max
  if real_max <= real_min or imag_max <= imag_min:
    raise ValueError(
      'region to cover must be prescribed with strict bounds: '
      '--real-max > --real-min and --imag-max > --imag-min')

  plot_root, plot_ext = os.path.splitext(args.plot_name)
  if plot_ext.lower() != '.html':
    args.plot_name = plot_root + '.html'
    print(f'adjusted plot-name to HTML output: {args.plot_name}', flush=True)

  jacobian_path = './lin_ops.h5'
  massmat_path = './mass_mat.h5'
  nr_local, keep = build_reduction_mapping(jacobian_path)

  os.makedirs(args.output_dir, exist_ok=True)
  write_run_metadata(
    args.output_dir,
    build_run_metadata(args, real_min, real_max, imag_min, imag_max))

  try:
    # Try to load precomputed reduced jacobian and mass matrices
    real_Jac = np.load('real_jacobian.npy')
    print(real_Jac.shape)
  except (FileNotFoundError, OSError, ValueError):
    dt = 1e-7
    # Get jacobian and mass matrices from files
    Mmat = Matrix(massmat_path, '/massmat')
    Jac = Matrix(jacobian_path, '/jacobian')

    Mmat_big = scipy.sparse.block_diag([Mmat.csr_rep] * 7, format='csr')
    del Mmat

    jac_gl = make_global_mat(Jac.csr_rep.toarray(), nrg=Jac.nrg, ncg=Jac.ncg, lg=Jac.lg)
    jac_gl = sparse.csr_array(jac_gl)
    jac_gl.eliminate_zeros()

    mmat_gl = make_global_mat(Mmat_big.toarray(), nrg=Jac.nrg, ncg=Jac.ncg, lg=Jac.lg)
    mmat_gl = sparse.csr_array(mmat_gl)
    mmat_gl.eliminate_zeros()

    print('Shape of jac_gl is:', jac_gl.shape, flush=True)
    print('shape of mmat_gl is:', mmat_gl.shape, flush=True)

    reduced_Jac = jac_gl[keep][:, keep].tocsr()
    iden = sparse.eye(reduced_Jac.shape[0], format='csr')
    reduced_Mmat = mmat_gl[keep][:, keep].tocsr()

    print('Shape of reduced_Jac is:', reduced_Jac.shape, flush=True)
    print('shape of reduced_Mmat is:', reduced_Mmat.shape, flush=True)
    # compute real jacobian
    real_Jac = sparse.linalg.spsolve(reduced_Jac, reduced_Mmat.toarray())
    del reduced_Jac, reduced_Mmat
    np.save('./real_jacobian.npy', real_Jac)

  # Compute full eigenvalue spectrum with LAPACK for plotting and diagnostics.
  try:
    w = np.load('./full_reduced_eigvals.npy')
  except FileNotFoundError:
    print("computing full eigenvalue spectrum with scipy.linalg.eig", flush=True)
    w = np.linalg.eigvals(real_Jac)
    w = (w-1)/dt
    _, v = sparse.linalg.eigs(real_Jac, k=40, ncv=90, which='LM')
    np.save('./full_reduced_eigvals.npy', w)
    np.save('./full_reduced_eigvecs.npy', v)
    os.makedirs('./eigvecs_plot', exist_ok=True)
    get_plot_vecs(v, keep, os.path.join(args.output_dir, 'eigvecs_plot'))
    plt.scatter(w.real, w.imag, s=2, c='k')
    plt.savefig('./full_reduced_spectrum.png')

  # Compute schur factorization
  # if full_reduced_schur.npy already exists, load it, otherwise compute it and save it to avoid recomputation in the future
  try:
    T = np.load('./full_reduced_schur.npy')
  except FileNotFoundError:
    os.makedirs('.', exist_ok=True)
    print("computing schur factorization", flush=True)
    T, _ = scipy.linalg.schur(real_Jac, output='complex')
    plt.scatter(T.diagonal().real, T.diagonal().imag, s=2, c='k')
    plt.savefig('./full_reduced_schur_eigs.png')
    np.save('./full_reduced_schur.npy', T)
  del real_Jac
  print('Running pseudospectrum', flush=True)
  R, C, sigmin = compute_pseudospectrum(
    T,
    grid_points=args.grid_points,
    nprocs=args.nprocs,
    real_min=real_min,
    real_max=real_max,
    imag_min=imag_min,
    imag_max=imag_max)
  del T
  # save grid and sigmin
  np.save(os.path.join(args.output_dir, 'pseudo_R.npy'), R)
  np.save(os.path.join(args.output_dir, 'pseudo_C.npy'), C)
  np.save(os.path.join(args.output_dir, 'pseudo_sigmin.npy'), sigmin)
  # save interactive contour plot with plotly
  plot_pseudospectrum(
    R,
    C,
    sigmin,
    output_path=os.path.join(args.output_dir, args.plot_name),
    auto_open=args.plot_auto_open,
    eig_marker_size=args.eig_marker_size,
    min_level=args.min_level,
    nlevels=args.nlevels,
    eigvals=w)