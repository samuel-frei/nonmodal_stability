"""End-to-end workflows, one per subcommand.

* `run_pipeline` -- reduce, factorise, sample, write out a pseudospectrum.
* `pseudomode_run` -- extract modes at given points, write them as restarts.
* `plot_run` -- render a finished run directory, without the operator.
"""

import os
import socket
from datetime import UTC, datetime
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray

from .adapters import find_input_deck, read_physical_scales
from .config import KreissConfig, PlotConfig, PseudomodeConfig, RunConfig
from .fields import build_reduction_mapping, write_restart_modes
from .io import (
  OPERATOR_DESCRIPTION,
  build_metadata,
  load_samples,
  save_samples,
  save_spectrum,
  write_eigenmodes,
  write_kreiss,
  write_metadata,
  write_pseudomodes,
)
from .kreiss import kreiss_constant
from .norm import Norm, scaled_l2_norm
from .operator import (
  load_or_compute_generator,
  load_or_compute_schur,
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
from .resolution import load_or_compute_resolution
from .sampling import Bounds, ResolvedSource, mirror_conjugates
from .subspace import Subspace, restrict_to_resolved, whole_operator


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

  generator = load_or_compute_generator(
    config.jacobian, config.massmat, keep_global, config.cache_dir, config.timestep)
  real_operator = bool(np.isrealobj(generator))

  # Every resolvent norm is the scaled L2 norm; R is rebuilt even on a cache
  # hit, since eigenvectors need it to return to coefficients.
  norm = scaled_l2_norm(config.jacobian, config.massmat, keep_global)

  # One factorisation serves all three: spectrum, eigenvectors, and sampling.
  schur_t, schur_z = load_or_compute_schur_vectors(generator, config.cache_dir, norm)
  del generator
  save_spectrum(config.output_dir, spectrum_from_schur(schur_t))
  # Everything below sees only the resolved modes, unless --all-modes.
  schur_t, schur_z, subspace = _restrict(config, schur_t, schur_z, norm, keep_global)
  # A real operator's spectrum is conjugate-symmetric, so half the plane serves --
  # unless a restriction split a pair.
  half_plane = real_operator and subspace.conjugate_closed and not config.force_full_plane
  eigvals = spectrum_from_schur(schur_t)
  _write_eigenmode_sidecar(config, *write_eigenmode_restarts(
    schur_t, schur_z, keep_global, nr_local, config.output_dir,
    config.n_eigvecs, norm))
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
    effective_workers=_worker_count(n_evaluated, config.nprocs),
    physical_scales=_physical_scales(config.jacobian),
    norm=norm.describe(),
    subspace=subspace.describe()))


_Compute = RunConfig | PseudomodeConfig | KreissConfig


def _resolution_cached(
  config: _Compute,
  schur_t: NDArray[np.complex128],
  norm: Norm,
  keep_global: NDArray[np.bool_],
) -> bool:
  """Whether the mode resolution for this `T` is already in the cache."""
  return load_or_compute_resolution(
    schur_t, None, norm, config.cache_dir, config.jacobian, config.massmat,
    config.stiffmat, keep_global) is not None


@overload
def _restrict(
  config: _Compute, schur_t: NDArray[np.complex128], schur_z: NDArray[np.complex128],
  norm: Norm, keep_global: NDArray[np.bool_],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], Subspace]: ...
@overload
def _restrict(
  config: _Compute, schur_t: NDArray[np.complex128], schur_z: None,
  norm: Norm, keep_global: NDArray[np.bool_],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128] | None, Subspace]: ...
def _restrict(
  config: _Compute,
  schur_t: NDArray[np.complex128],
  schur_z: NDArray[np.complex128] | None,
  norm: Norm,
  keep_global: NDArray[np.bool_],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128] | None, Subspace]:
  """Restrict to the resolved modes, or pass everything through for --all-modes."""
  if config.all_modes:
    return schur_t, schur_z, whole_operator(schur_t)
  resolution = load_or_compute_resolution(
    schur_t, schur_z, norm, config.cache_dir, config.jacobian, config.massmat,
    config.stiffmat, keep_global)
  if resolution is None:
    raise RuntimeError('the mode resolution needs the Schur vectors, which were not loaded')
  return restrict_to_resolved(
    schur_t, schur_z, resolution, config.resolution_cutoff, config.stiffmat)


def _physical_scales(jacobian_path: str) -> dict[str, Any]:
  """The deck's reference scales, logged and returned for a metadata sidecar."""
  scales = read_physical_scales(find_input_deck(jacobian_path))
  print(f'tau_A = {scales.alfven_time_s:.4g} s, S = {scales.lundquist_number:.4g}, '
        f'Pm = {scales.magnetic_prandtl_number:.4g} ({scales.unit_convention})', flush=True)
  return scales.describe()


def pseudomode_run(config: PseudomodeConfig) -> list[str]:
  """Extract pseudomodes at the given points and write them as restart files."""
  nr_local, keep_global = build_reduction_mapping(config.jacobian)
  os.makedirs(config.output_dir, exist_ok=True)

  generator = load_or_compute_generator(
    config.jacobian, config.massmat, keep_global, config.cache_dir, config.timestep)
  norm = scaled_l2_norm(config.jacobian, config.massmat, keep_global)
  schur_t, schur_z = load_or_compute_schur_vectors(generator, config.cache_dir, norm)
  del generator
  schur_t, schur_z, subspace = _restrict(config, schur_t, schur_z, norm, keep_global)

  modes = [pseudomode_at(schur_t, schur_z, z, config.tol, norm) for z in config.points]
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
    'physical_scales': _physical_scales(config.jacobian),
    'norm': norm.describe(),
    'subspace': subspace.describe(),
  })
  return paths


def kreiss_run(config: KreissConfig) -> dict[str, Any]:
  """Bound the Kreiss constant of the generator and write `kreiss.json`."""
  _, keep_global = build_reduction_mapping(config.jacobian)
  generator = load_or_compute_generator(
    config.jacobian, config.massmat, keep_global, config.cache_dir, config.timestep)
  norm = scaled_l2_norm(config.jacobian, config.massmat, keep_global)
  # The search needs only T. Z is loaded only when the mode resolution is not
  # yet cached, since that is computed from the eigenvectors.
  schur_t = load_or_compute_schur(generator, config.cache_dir, norm)
  schur_z = None
  if not config.all_modes and not _resolution_cached(config, schur_t, norm, keep_global):
    schur_t, schur_z = load_or_compute_schur_vectors(generator, config.cache_dir, norm)
  del generator
  save_spectrum(config.output_dir, spectrum_from_schur(schur_t))
  schur_t, _, subspace = _restrict(config, schur_t, schur_z, norm, keep_global)
  del schur_z
  if not subspace.conjugate_closed:
    print('warning: the off-axis check scans Im z > 0 only, which assumes a '
          'conjugate-symmetric spectrum', flush=True)

  result = kreiss_constant(
    schur_t, config.nprocs, axis_points=config.axis_points,
    offaxis_x=config.offaxis_x, offaxis_y=config.offaxis_y, tol=config.tol)
  del schur_t

  scales = _physical_scales(config.jacobian)
  tau_a = float(scales['alfven_time_s'])
  payload: dict[str, Any] = {
    'created_utc': datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z'),
    'hostname': socket.gethostname(),
    'cwd': os.getcwd(),
    'slurm_job_id': os.environ.get('SLURM_JOB_ID', ''),
    'run_tag': config.run_tag,
    'case_tag': config.case_tag,
    'inputs': {
      'jacobian': os.path.abspath(config.jacobian),
      'massmat': os.path.abspath(config.massmat),
      'cache_dir': os.path.abspath(config.cache_dir),
    },
    'operator': OPERATOR_DESCRIPTION,
    'norm': norm.describe(),
    'subspace': subspace.describe(),
    'physical_scales': scales,
    'z_star_times_tau_A': [result.z_star_real * tau_a, result.z_star_imag * tau_a],
    'spectral_abscissa_times_tau_A': result.spectral_abscissa * tau_a,
    'result': result.describe(),
  }
  write_kreiss(config.output_dir, payload)

  where = 'on the real axis' if not result.offaxis_exceeds_axis else 'OFF the real axis'
  print(f'Kreiss lower bound K = {result.kreiss:.6g} at z* = '
        f'{complex(result.z_star_real, result.z_star_imag):.6g} ({where}); '
        f'sup_t ||e^(tL)|| in [{result.kreiss:.4g}, {result.upper_bound:.4g}]', flush=True)
  if not result.axis_is_local_max_in_y:
    print('warning: the real-axis maximiser is a saddle in Im z, not a maximum', flush=True)
  if result.axis_at_upper_end:
    print('note: the real-axis maximum is at the scan\'s upper end; K -> 1 there, as for '
          'a (nearly) normal operator', flush=True)
  return payload


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
