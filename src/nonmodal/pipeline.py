"""End-to-end workflows: sample an operator, and render a finished run."""

import os

import numpy as np
from numpy.typing import NDArray

from .config import PlotConfig, RunConfig
from .fields import build_reduction_mapping
from .io import build_metadata, load_samples, save_samples, write_metadata
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

  Kept separate from `run_pipeline` so the Schur factor -- the largest array in
  the run -- can be released as soon as sampling is done.
  """
  points = source.build()
  if half_plane:
    points = points[points.imag >= 0.0]

  sigmin = sample_sigmin(points, schur_t, config.nprocs)
  if config.refine_points < 1:
    return points, sigmin

  def sample(batch: NDArray[np.complex128]) -> NDArray[np.float64]:
    return sample_sigmin(batch, schur_t, config.nprocs, 'refinement points')

  # refine_points counts evaluations added on top of whatever the initial grid
  # cost, so half-plane sampling needs no adjustment here.
  return refine(
    points, sigmin, sample,
    budget=points.size + config.refine_points,
    rounds=config.refine_rounds)


def run_pipeline(config: RunConfig) -> None:
  """Reduce, factorise, sample, and write out a pseudospectrum run."""
  nr_local, keep_global = build_reduction_mapping(config.jacobian)
  os.makedirs(config.output_dir, exist_ok=True)
  os.makedirs(config.cache_dir, exist_ok=True)

  real_jac = load_or_compute_jacobian(
    config.jacobian, config.massmat, keep_global, config.cache_dir, config.timestep)
  eigvals = load_or_compute_eigvals(
    real_jac, keep_global, nr_local, config.output_dir, config.cache_dir,
    config.n_eigvecs)
  # A real operator has a conjugate-symmetric spectrum, so the lower half-plane
  # is redundant. This is the actual mathematical precondition, checked on the
  # operator rather than inferred from the shape of the sample set.
  half_plane = bool(np.isrealobj(real_jac)) and not config.force_full_plane
  schur_t = load_or_compute_schur(real_jac, config.cache_dir)
  del real_jac

  # Bounds may be inferred from the spectrum, so the source is only concrete
  # once the eigenvalues exist.
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
