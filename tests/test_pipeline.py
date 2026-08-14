"""Sampling flow: coarse initial grid, then refinement on top of it."""

import numpy as np
import scipy.linalg

from nonmodal.config import RunConfig
from nonmodal.pipeline import _sample
from nonmodal.sampling import SpectrumSource


def _real_operator(n: int = 12) -> tuple[np.ndarray, np.ndarray]:
  """A real matrix and its Schur factor, as run_pipeline would produce."""
  rng = np.random.default_rng(2)
  A = rng.normal(size=(n, n))
  eigvals = np.linalg.eigvals(A)
  schur_t = np.asarray(scipy.linalg.schur(A, output='complex')[0], dtype=np.complex128)
  return eigvals, schur_t


def _resolved(eigvals: np.ndarray, nx: int, ny: int):
  return SpectrumSource(nx=nx, ny=ny).resolve(eigvals)


def test_without_refinement_only_the_initial_grid_is_sampled() -> None:
  eigvals, T = _real_operator()
  config = RunConfig(source=SpectrumSource(nx=6, ny=6), nprocs=2)

  points, sigmin = _sample(config, _resolved(eigvals, 6, 6), T, half_plane=False)

  assert points.size == 36
  assert sigmin.size == points.size


def test_refine_points_are_spent_on_top_of_the_initial_grid() -> None:
  """refine_points counts *extra* evaluations, not a total budget."""
  eigvals, T = _real_operator()
  config = RunConfig(
    source=SpectrumSource(nx=6, ny=6), nprocs=2,
    refine_points=40, refine_rounds=2)

  points, sigmin = _sample(config, _resolved(eigvals, 6, 6), T, half_plane=False)

  assert 36 < points.size <= 36 + 40
  assert sigmin.size == points.size


def test_half_plane_does_not_shrink_the_refinement_budget() -> None:
  """The initial grid halves, but refine_points still buys what it says."""
  eigvals, T = _real_operator()
  config = RunConfig(
    source=SpectrumSource(nx=8, ny=8), nprocs=2,
    refine_points=30, refine_rounds=2)

  resolved = _resolved(eigvals, 8, 8)
  points, _ = _sample(config, resolved, T, half_plane=True)

  # Half-plane sampling keeps only Im z >= 0 rows before refinement starts.
  seeded = int(np.count_nonzero(resolved.build().imag >= 0.0))
  assert np.all(points.imag >= 0.0), 'half-plane sampling must stay above the axis'
  assert seeded < points.size <= seeded + 30
