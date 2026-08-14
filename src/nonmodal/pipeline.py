"""End-to-end workflows, one per subcommand.

* `run_pipeline` -- reduce, factorise, sample, write out a pseudospectrum.
* `pseudomode_run` -- extract modes at given points, write them as restarts.
* `plot_run` -- render a finished run directory, without the operator.
"""

import os

import numpy as np
from numpy.typing import NDArray

from .config import PlotConfig, PseudomodeConfig, RunConfig
from .fields import build_reduction_mapping, write_restart_modes
from .io import (
  build_metadata,
  load_samples,
  save_samples,
  write_eigenmodes,
  write_metadata,
  write_pseudomodes,
)
from .operator import (
  load_or_compute_jacobian,
  load_or_compute_schur_vectors,
  spectrum_from_schur,
  write_eigenmode_restarts,
)
from .plotting import (
  _normalize_html_name,
  _split_plot_output_names,
  pseudo_contours,
  pseudo_heatmap,
)
from .pseudomode import pseudomode_at
from .pseudospectrum import _worker_count, choose_contour_levels, sample_sigmin
from .refine import refine
from .sampling import Bounds, ResolvedSource, mirror_conjugates


def _sample(
  config: RunConfig,
  source: ResolvedSource,
  schur_t: NDArray[np.complexfloating],
  half_plane: bool,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
  """Evaluate the coarse initial grid, then refine onto features if asked.

  Separate so the Schur factor is released as soon as sampling finishes.
  """
  points = source.build()
  if half_plane:
    points = points[points.imag >= 0.0]

  sigmin = sample_sigmin(points, schur_t, config.nprocs)
  if config.refine_points < 1:
    return points, sigmin

  def sample(batch: NDArray[np.complex128]) -> NDArray[np.float64]:
    return sample_sigmin(batch, schur_t, config.nprocs, 'refinement points')

  # refine_points counts evaluations added on top of the initial grid.
  return refine(
    points, sigmin, sample,
    budget=points.size + config.refine_points,
    rounds=config.refine_rounds)


def _write_eigenmode_sidecar(
  config: RunConfig,
  chosen: NDArray[np.complex128],
  paths: list[str],
) -> None:
  """Record which eigenvalue each restart file holds.

  A conjugate pair contributes two entries whose `Re(v)` restarts are identical.
  """
  eigvec_dir = os.path.join(config.output_dir, 'eigvecs_plot')
  write_eigenmodes(eigvec_dir, {
    'run_tag': config.run_tag,
    'case_tag': config.case_tag,
    'modes': [
      {
        'eigenvalue_real': float(lam.real),
        'eigenvalue_imag': float(lam.imag),
        'file': os.path.relpath(path, eigvec_dir),
      }
      for lam, path in zip(chosen, paths, strict=True)
    ],
  })


def run_pipeline(config: RunConfig) -> None:
  """Reduce, factorise, sample, and write out a pseudospectrum run."""
  nr_local, keep_global = build_reduction_mapping(config.jacobian)
  os.makedirs(config.output_dir, exist_ok=True)
  os.makedirs(config.cache_dir, exist_ok=True)

  real_jac = load_or_compute_jacobian(
    config.jacobian, config.massmat, keep_global, config.cache_dir, config.timestep)
  # A real operator's spectrum is conjugate-symmetric, so half the plane serves.
  real_operator = bool(np.isrealobj(real_jac))
  half_plane = real_operator and not config.force_full_plane

  # One factorisation serves all three: spectrum, eigenvectors, and sampling.
  schur_t, schur_z = load_or_compute_schur_vectors(real_jac, config.cache_dir)
  del real_jac
  eigvals = spectrum_from_schur(schur_t, config.cache_dir)
  _write_eigenmode_sidecar(config, *write_eigenmode_restarts(
    schur_t, schur_z, keep_global, nr_local, config.output_dir,
    config.cache_dir, config.n_eigvecs))
  # Z is the same size as T and sampling never touches it.
  del schur_z

  # Bounds may come from the spectrum, so the source is concrete only now.
  source = config.source.resolve(eigvals)
  points, sigmin = _sample(config, source, schur_t, half_plane)
  del schur_t

  n_evaluated = int(points.size)
  if half_plane:
    # Everything sampled lives in Im z >= 0; the rest follows by conjugation.
    points, sigmin = mirror_conjugates(points, sigmin)

  save_samples(config.output_dir, points, sigmin, eigvals)
  write_metadata(config.output_dir, build_metadata(
    config,
    Bounds.around_points(points),
    n_points=int(points.size),
    n_evaluated=n_evaluated,
    half_plane=half_plane,
    effective_workers=_worker_count(n_evaluated, config.nprocs)))


def pseudomode_run(config: PseudomodeConfig) -> list[str]:
  """Extract pseudomodes at the given points and write them as restart files."""
  nr_local, keep_global = build_reduction_mapping(config.jacobian)
  os.makedirs(config.output_dir, exist_ok=True)

  real_jac = load_or_compute_jacobian(
    config.jacobian, config.massmat, keep_global, config.cache_dir, config.timestep)
  schur_t, schur_z = load_or_compute_schur_vectors(real_jac, config.cache_dir)
  del real_jac

  modes = [pseudomode_at(schur_t, schur_z, z, config.tol) for z in config.points]
  del schur_t, schur_z

  mode_dir = os.path.join(config.output_dir, config.mode_dir)
  paths = write_restart_modes(
    np.column_stack([m.vector for m in modes]),
    keep_global, nr_local, mode_dir, phases=config.phases)
  print(f'wrote {len(paths)} restart files to {os.path.abspath(mode_dir)}', flush=True)

  write_pseudomodes(config.output_dir, {
    'run_tag': config.run_tag,
    'case_tag': config.case_tag,
    'phases': int(config.phases),
    'tol': float(config.tol),
    'modes': [m.describe() for m in modes],
    'files': [os.path.relpath(p, config.output_dir) for p in paths],
  })
  return paths


def plot_run(config: PlotConfig) -> tuple[str, str]:
  """Render a finished run directory, without touching the operator."""
  z, sigmin, eigvals = load_samples(config.output_dir)
  levels = (
    np.asarray(config.levels, dtype=float)
    if config.levels
    else choose_contour_levels(
      sigmin, min_level=config.min_level, nlevels=config.nlevels))

  print(
    f'plotting {z.size} samples, levels='
    f'{np.array2string(np.asarray(levels), precision=3)}', flush=True)

  plot_name = _normalize_html_name(config.plot_name, 'plot-name')
  heatmap_name, contour_name = _split_plot_output_names(plot_name)
  heatmap = pseudo_heatmap(
    config.output_dir, heatmap_name, z, sigmin, eigvals,
    mesh=config.mesh, inline_js=config.inline_js)
  contours = pseudo_contours(
    config.output_dir, contour_name, z, sigmin, eigvals, levels,
    inline_js=config.inline_js)
  return heatmap, contours
