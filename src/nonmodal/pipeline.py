"""End-to-end reduced pseudospectrum workflow."""

import argparse
import os

import numpy as np

from .fields import build_reduction_mapping
from .grid import grid_bounds_from_flat, load_flat_grid_npy
from .io import (
  build_metadata,
  save_pseudospectrum_arrays,
  save_pseudospectrum_flat,
  write_metadata,
)
from .operator import (
  load_or_compute_eigvals,
  load_or_compute_jacobian,
  load_or_compute_schur,
)
from .plotting import (
  _normalize_html_name,
  _split_plot_output_names,
  pseudo_contours,
  pseudo_heatmap,
)
from .pseudospectrum import (
  _compute_sigmin_points,
  choose_contour_levels,
  compute_pseudospectrum,
)


def validate_and_normalize_args(
  args: argparse.Namespace,
) -> tuple[float | None, float | None, float | None, float | None]:
  """Validate pipeline inputs and normalise output naming."""
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


def run_pipeline(args: argparse.Namespace) -> None:
  """Run the end-to-end reduced pseudospectrum workflow."""
  real_min, real_max, imag_min, imag_max = validate_and_normalize_args(args)

  jacobian_path = args.jacobian
  massmat_path = args.massmat
  cache_dir = args.cache_dir
  nr_local, keep_global = build_reduction_mapping(jacobian_path)

  os.makedirs(args.output_dir, exist_ok=True)
  os.makedirs(cache_dir, exist_ok=True)

  real_jac = load_or_compute_jacobian(jacobian_path, massmat_path, keep_global, cache_dir)
  eigvals = load_or_compute_eigvals(
    real_jac, keep_global, nr_local, args.output_dir, cache_dir)
  schur_t = load_or_compute_schur(real_jac, cache_dir)
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

  # Both branches above have resolved the bounds: the --grid-npy branch derives
  # them from the loaded grid, and the generated branch had them validated as
  # non-None before sampling.
  assert real_min is not None and real_max is not None
  assert imag_min is not None and imag_max is not None

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
    # plot_enabled is only True on paths that produced a structured R/C mesh.
    assert R is not None and C is not None
    levels = choose_contour_levels(sigmin, min_level=args.min_level, nlevels=args.nlevels)
    print(
      f'plotting levels={np.array2string(levels, precision=3)}, '
      f'xlim=({real_min:.6g}, {real_max:.6g}), '
      f'ylim=({imag_min:.6g}, {imag_max:.6g})',
      flush=True)

    heatmap_plot_name, contour_plot_name = _split_plot_output_names(args.plot_name)
    pseudo_heatmap(args.output_dir, heatmap_plot_name, R, C, sigmin, eigvals)
    pseudo_contours(args.output_dir, contour_plot_name, R, C, sigmin, eigvals, levels)

    metadata['levels']['values'] = [float(v) for v in np.asarray(levels, dtype=float).ravel()]
    metadata['plot'] = {
      'enabled': True,
      'plot_name': args.plot_name,
      'plot_name_heatmap': heatmap_plot_name,
      'plot_name_contours': contour_plot_name,
    }
  else:
    metadata['plot'] = {
      'enabled': False,
      'reason': 'unstructured grid; provide --grid-shape to enable plots',
    }

  write_metadata(args.output_dir, metadata)
  if plot_enabled:
    assert R is not None and C is not None
    save_pseudospectrum_arrays(args.output_dir, R, C, sigmin)
  else:
    save_pseudospectrum_flat(args.output_dir, zz_flat, sigmin)
