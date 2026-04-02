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
from matplotlib import colors as mcolors
from numba import njit
import multiprocessing
from multiprocessing import shared_memory

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
_worker_I = None
_worker_trtrs = None
_shm = None
# Initialize worker process by attaching to shared memory and prebuilding helpers.
def _init_worker(shm_name, T_shape, T_dtype):
  global _worker_T, _worker_I, _worker_trtrs, _shm
  _shm = shared_memory.SharedMemory(name=shm_name)
  _worker_T = np.ndarray(T_shape, dtype=T_dtype, buffer=_shm.buf)
  _worker_I = np.eye(T_shape[0], dtype=T_dtype)
  _worker_trtrs, = scipy.linalg.get_lapack_funcs(('trtrs',), (_worker_T,))

# Compute the smallest singular value of zI - T using the Schur factorization and the trtrs LAPACK routine for triangular solves.
# This is done by defining a linear operator that applies the inverse of zI - T to a vector, and then using sparse.linalg.eigsh
# to compute the largest eigenvalue of this operator, which corresponds to the smallest singular value of zI - T.
def _compute_sig_for_z(z):
  trtrs = _worker_trtrs
  T = _worker_T
  I = _worker_I
  T1 = z*I - T

  def _matvec(q):
    tmp, _ = trtrs(T1, q.reshape(-1, 1), lower=0, trans=2, unitdiag=0)
    result, _ = trtrs(T1, tmp, lower=0, trans=0, unitdiag=0)
    return result.ravel()
  
  op = scipy.sparse.linalg.LinearOperator(T1.shape, matvec=_matvec, dtype=np.complex128)
  vals, _ = sparse.linalg.eigsh(op, k=1, which='LM', ncv=10, tol=1e-8)
  sig_min = vals[0]
  return 1/np.sqrt(sig_min)

def _compute_sig_block(args):
  i_start, z_block = args
  block_sig = np.zeros((z_block.shape[0],))
  for i in range(z_block.shape[0]):
    block_sig[i] = _compute_sig_for_z(z_block[i])
  return i_start, block_sig

def _build_work_blocks(zz_flat, worker_count, block_factor):
  """Create many small blocks so workers can pull work dynamically."""
  npts = int(zz_flat.shape[0])
  factor = max(1, int(block_factor))
  target_blocks = min(npts, worker_count * factor)
  block_size = max(1, npts // target_blocks)
  block_items = []
  for i_start in range(0, npts, block_size):
    i_end = min(i_start + block_size, npts)
    block_items.append((i_start, zz_flat[i_start:i_end]))
  return block_items

def _effective_worker_count(total_points, requested_nprocs):
  """Use as many workers as requested, up to one worker per work item."""
  return max(1, min(int(requested_nprocs), int(total_points)))

def _effective_grid_points(total_points, worker_count):
  """Round total points up so each worker receives the same number of items."""
  n = int(total_points)
  w = int(worker_count)
  return ((n + w - 1) // w) * w

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
  effective_points = _effective_grid_points(args.grid_points, effective_workers)
  rows, cols = _grid_shape(effective_points)
  return {
    'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
    'hostname': socket.gethostname(),
    'cwd': os.getcwd(),
    'run_tag': args.run_tag,
    'case_tag': args.case_tag,
    'slurm_job_id': os.environ.get('SLURM_JOB_ID', ''),
    'grid_points': int(effective_points),
    'grid_points_requested': int(args.grid_points),
    'grid_points_effective': int(effective_points),
    'grid_shape': {
      'rows': int(rows),
      'cols': int(cols),
    },
    'nprocs_requested': int(args.nprocs),
    'nprocs_effective': int(effective_workers),
    'block_factor': int(args.block_factor),
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
      'surface3d_name': args.surface3d_name,
    },
  }

def write_run_metadata(output_dir, metadata):
  path = os.path.join(output_dir, 'run_metadata.json')
  with open(path, 'w', encoding='ascii') as f:
    json.dump(metadata, f, indent=2, sort_keys=True)
  print(f'wrote metadata: {path}', flush=True)

def compute_full_spectrum_scipy(mat):
  """Compute the full eigenvalue spectrum with scipy.linalg.eig."""
  a = np.asarray(mat)
  w = scipy.linalg.eig(a, left=False, right=False, overwrite_a=False, check_finite=False)
  return np.asarray(w)

# Compute the pseudospectrum by iterating over a grid of points 
# in the complex plane, and for each point, 
# computing the smallest singular value of zI - T using the 
# _compute_sig_for_z function.
def compute_pseudospectrum(
  imat,
  grid_points=128,
  nprocs=10,
  block_factor=8,
  real_min=None,
  real_max=None,
  imag_min=None,
  imag_max=None):
  T = imat

  shm = shared_memory.SharedMemory(create=True, size=T.nbytes)
  T_shm = np.ndarray(T.shape, dtype=T.dtype, buffer=shm.buf)
  np.copyto(T_shm, T)
  
  if real_min is None or real_max is None or imag_min is None or imag_max is None:
    raise ValueError('real/imag bounds must all be provided')

  if real_max <= real_min:
    raise ValueError('real_max must be greater than real_min')
  if imag_max <= imag_min:
    raise ValueError('imag_max must be greater than imag_min')

  worker_count = _effective_worker_count(grid_points, nprocs)
  effective_points = _effective_grid_points(grid_points, worker_count)
  if effective_points != int(grid_points):
    print(
      f'adjusted grid_points from {int(grid_points)} to {effective_points} '
      f'to evenly distribute work across {worker_count} workers',
      flush=True)

  nrows, ncols = _grid_shape(effective_points)

  r = np.linspace(real_min, real_max, ncols)
  c = np.linspace(imag_min, imag_max, nrows)
  R, C =  np.meshgrid(r, c)
  zz = R + 1j * C
  sigmin_flat = np.zeros((effective_points,))
  print(
    f"computing pseudospectrum on Re[z] in [{real_min:.6g}, {real_max:.6g}] "
    f"and Im[z] in [{imag_min:.6g}, {imag_max:.6g}] with shape=({nrows},{ncols})",
    flush=True)

  zz_flat = zz.ravel()
  block_items = _build_work_blocks(zz_flat, worker_count, block_factor)

  print(
    f"using nprocs={worker_count}, block_factor={int(block_factor)}, point_blocks={len(block_items)}, "
    f"work_items={nrows * ncols}",
    flush=True)
  try:
    ctx = multiprocessing.get_context('forkserver')
    with ctx.Pool(
      processes=worker_count,
      initializer=_init_worker, 
      initargs=(shm.name, T.shape, T.dtype)) as pool:
      for i_start, block_sig in pool.imap_unordered(_compute_sig_block, block_items, chunksize=1):
        blen = block_sig.shape[0]
        sigmin_flat[i_start:i_start+blen] = block_sig
  finally:
    shm.close()
    shm.unlink()
  sigmin = sigmin_flat.reshape(nrows, ncols)
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

  # Keep level selection simple and deterministic: use geometric spacing
  # across the finite positive data range, with a floor at min_level.
  lo = max(min_level, data_min)
  hi = data_max
  if hi <= lo:
    # Matplotlib contour colorbars require at least two boundaries.
    return np.array([lo, np.nextafter(lo, np.inf)])
  return np.geomspace(lo, hi, nlevels)

def plot_pseudospectrum(
  R,
  C,
  sigmin,
  output_path,
  eigvals,
  min_level=1e-12,
  nlevels=5):
  levels = choose_contour_levels(sigmin, min_level=min_level, nlevels=nlevels)
  print(
    f'plot_pseudospectrum: levels={np.array2string(levels, precision=3)}, '
    f'xlim=({float(np.min(R)):.6g}, {float(np.max(R)):.6g}), '
    f'ylim=({float(np.min(C)):.6g}, {float(np.max(C)):.6g})',
    flush=True)
  fig, ax = plt.subplots()
  artist = ax.contour(R, C, sigmin, levels=levels)

  vals = np.asarray(eigvals)
  vals = vals[np.isfinite(vals)]
  if vals.size > 0:
    ax.scatter(vals.real, vals.imag, s=12, c='k', alpha=0.55, linewidths=0, zorder=3)

  ax.set_xlim(float(np.min(R)), float(np.max(R)))
  ax.set_ylim(float(np.min(C)), float(np.max(C)))
  lo = float(levels[0])
  hi = float(levels[-1])
  if hi <= lo:
    hi = np.nextafter(lo, np.inf)
  sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=lo, vmax=hi), cmap=artist.cmap)
  sm.set_array([])
  plt.colorbar(sm, ax=ax)
  ax.set_title('Pseudospectrum')
  fig.savefig(output_path, format='png')
  plt.close(fig)

def save_sigmin_surface_html(
  R,
  C,
  sigmin,
  output_path,
  eigvals=None,
  auto_open=False):
  try:
    import plotly.graph_objects as go
  except ImportError as exc:
    raise ImportError('plotly is required for interactive 3D surface export (pip install plotly)') from exc

  zvals = np.asarray(sigmin, dtype=float)
  fig = go.Figure()
  fig.add_trace(go.Surface(
    x=np.asarray(R, dtype=float),
    y=np.asarray(C, dtype=float),
    z=zvals,
    colorscale='Viridis',
    colorbar=dict(title='sigmin')))

  if eigvals is not None:
    vals = np.asarray(eigvals)
    vals = vals[np.isfinite(vals)]
    if vals.size > 0:
      z_floor = float(np.nanmin(zvals[np.isfinite(zvals)])) if np.any(np.isfinite(zvals)) else 0.0
      fig.add_trace(go.Scatter3d(
        x=vals.real,
        y=vals.imag,
        z=np.full(vals.shape, z_floor),
        mode='markers',
        marker=dict(size=2, color='black', opacity=0.7),
        name='eigenvalues'))

  fig.update_layout(
    title='Pseudospectrum sigmin surface',
    scene=dict(
      xaxis_title='Re[z]',
      yaxis_title='Im[z]',
      zaxis_title='sigmin'),
    template='plotly_white',
    margin=dict(l=0, r=0, b=0, t=40))

  fig.write_html(output_path, include_plotlyjs='cdn', full_html=True, auto_open=auto_open)
  print(f'wrote interactive 3D surface: {output_path}', flush=True)

if __name__=="__main__":
  parser = argparse.ArgumentParser(description='Compute reduced pseudospectrum and save outputs.')
  parser.add_argument('--grid-points', type=int, default=128, help='Total number of grid points (minimum 128).')
  parser.add_argument('--nprocs', type=int, default=128, help='Worker process count.')
  parser.add_argument('--block-factor', type=int, default=8,
                      help='Multiplier controlling number of point-blocks per worker for dynamic scheduling.')
  parser.add_argument('--real-center', type=float, default=0.0,
                      help='Center of sampled real-axis window when --real-min/--real-max are not set.')
  parser.add_argument('--imag-center', type=float, default=0.0,
                      help='Center of sampled imaginary-axis window when --imag-min/--imag-max are not set.')
  parser.add_argument('--real-half-width', type=float, default=1e-3,
                      help='Half-width of sampled real-axis window about --real-center.')
  parser.add_argument('--imag-half-width', type=float, default=1e-3,
                      help='Half-width of sampled imaginary-axis window about --imag-center.')
  parser.add_argument('--real-min', type=float, default=None,
                      help='Explicit sampled real-axis minimum; overrides center/half-width if set with --real-max.')
  parser.add_argument('--real-max', type=float, default=None,
                      help='Explicit sampled real-axis maximum; overrides center/half-width if set with --real-min.')
  parser.add_argument('--imag-min', type=float, default=None,
                      help='Explicit sampled imaginary-axis minimum; overrides center/half-width if set with --imag-max.')
  parser.add_argument('--imag-max', type=float, default=None,
                      help='Explicit sampled imaginary-axis maximum; overrides center/half-width if set with --imag-min.')
  parser.add_argument('--min-level', type=float, default=1e-5, help='Minimum contour level.')
  parser.add_argument('--nlevels', type=int, default=5, help='Number of decade-spaced contour levels.')
  parser.add_argument('--run-tag', type=str, default='', help='Batch-level run identifier for metadata tracking.')
  parser.add_argument('--case-tag', type=str, default='', help='Case identifier for metadata tracking.')
  parser.add_argument('--output-dir', type=str, default='reduced', help='Output directory for arrays and plot.')
  parser.add_argument('--plot-name', type=str, default='pseudoplot.png', help='Output plot filename.')
  parser.add_argument('--surface3d-name', type=str, default='sigmin_surface.html',
                      help='Interactive 3D surface output filename (HTML).')
  parser.add_argument('--surface3d-auto-open', action='store_true',
                      help='Attempt to open the interactive 3D surface in a browser after writing it.')
  args = parser.parse_args()

  if args.grid_points < 128:
    raise ValueError('grid-points must be >= 128')
  if args.nprocs < 1:
    raise ValueError('nprocs must be >= 1')
  if args.block_factor < 1:
    raise ValueError('block-factor must be >= 1')

  if (args.real_min is None) ^ (args.real_max is None):
    raise ValueError('provide both --real-min and --real-max, or neither')
  if (args.imag_min is None) ^ (args.imag_max is None):
    raise ValueError('provide both --imag-min and --imag-max, or neither')

  real_half = args.real_half_width
  imag_half = args.imag_half_width
  if real_half <= 0.0:
    raise ValueError('real-half-width must be positive')
  if imag_half <= 0.0:
    raise ValueError('imag-half-width must be positive')

  if args.real_min is None:
    real_min = args.real_center - real_half
    real_max = args.real_center + real_half
  else:
    real_min = args.real_min
    real_max = args.real_max

  if args.imag_min is None:
    imag_min = args.imag_center - imag_half
    imag_max = args.imag_center + imag_half
  else:
    imag_min = args.imag_min
    imag_max = args.imag_max

  os.makedirs(args.output_dir, exist_ok=True)
  write_run_metadata(
    args.output_dir,
    build_run_metadata(args, real_min, real_max, imag_min, imag_max))

  try:
    # Try to load precomputed reduced jacobian and mass matrices
    real_Jac = np.load('reduced/real_jacobian.npy')
  except (FileNotFoundError, OSError, ValueError):
    # Get jacobian and mass matrices from files
    Mmat = Matrix('/ocean/projects/phy240045p/freiberg/pseudospectra/harris_linear/mass_mat.h5', '/massmat')
    Jac = Matrix('/ocean/projects/phy240045p/freiberg/pseudospectra/harris_linear/lin_ops.h5', '/jacobian')

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

    # Boolean block masking: keep selected state blocks by index.
    nrg_block = Jac.nrg // 7
    block_mask = np.zeros(Jac.nrg, dtype=bool)
    for i in (0, 1, 3, 4, 5):
      block_mask[nrg_block * i:nrg_block * (i + 1)] = True

    keep = block_mask

    # If bc flags are present, also remove constrained rows/columns.
    if np.any(Jac.bcg):
      keep &= ~Jac.bcg

    reduced_Jac = jac_gl[keep][:, keep].tocsr()
    reduced_Mmat = mmat_gl[keep][:, keep].tocsr()

    print('Shape of jac_gl_reduced is:', reduced_Jac.shape, flush=True)
    print('shape of mmat_gl_reduced is:', reduced_Mmat.shape, flush=True)

    # compute real jacobian
    real_Jac = sparse.linalg.spsolve(reduced_Jac, reduced_Mmat.toarray())
    del reduced_Jac, reduced_Mmat
    os.makedirs('reduced', exist_ok=True)
    np.save('reduced/real_jacobian.npy', real_Jac)

  # Compute full eigenvalue spectrum with LAPACK for plotting and diagnostics.
  try:
    w = np.load('reduced/full_reduced_eigs_scipy.npy')
  except FileNotFoundError:
    os.makedirs('reduced', exist_ok=True)
    print("computing full eigenvalue spectrum with scipy.linalg.eig", flush=True)
    w = compute_full_spectrum_scipy(real_Jac)
    np.save('reduced/full_reduced_eigs_scipy.npy', w)
    plt.scatter(w.real, w.imag)
    plt.savefig('reduced/full_reduced_spectrum_scipy.png')

  # Compute schur factorization
  # if full_reduced_schur.npy already exists, load it, otherwise compute it and save it to avoid recomputation in the future
  try:
    T = np.load('reduced/full_reduced_schur.npy')
  except FileNotFoundError:
    os.makedirs('reduced', exist_ok=True)
    print("computing schur factorization", flush=True)
    T, _ = scipy.linalg.schur(real_Jac, output='complex')
    plt.scatter(T.diagonal().real, T.diagonal().imag)
    plt.savefig('reduced/full_reduced_schur_eigs.png')
    np.save('reduced/full_reduced_schur.npy', T)
  del real_Jac
  print('Running pseudospectrum', flush=True)
  R, C, sigmin = compute_pseudospectrum(
    T,
    grid_points=args.grid_points,
    nprocs=args.nprocs,
    block_factor=args.block_factor,
    real_min=real_min,
    real_max=real_max,
    imag_min=imag_min,
    imag_max=imag_max)
  del T
  # save grid and sigmin
  np.save(os.path.join(args.output_dir, 'pseudo_R.npy'), R)
  np.save(os.path.join(args.output_dir, 'pseudo_C.npy'), C)
  np.save(os.path.join(args.output_dir, 'pseudo_sigmin.npy'), sigmin)
  plot_pseudospectrum(
    R,
    C,
    sigmin,
    output_path=os.path.join(args.output_dir, args.plot_name),
    min_level=args.min_level,
    nlevels=args.nlevels,
    eigvals=w)
  surface_path = os.path.join(args.output_dir, args.surface3d_name)
  save_sigmin_surface_html(
    R,
    C,
    sigmin,
    output_path=surface_path,
    eigvals=w,
    auto_open=args.surface3d_auto_open)