"""Adaptive refinement: the error indicator, the budget, and degenerate input."""

import numpy as np
import pytest
import scipy.linalg
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from nonmodal.pseudospectrum import sample_sigmin
from nonmodal.refine import refine, triangle_errors
from nonmodal.sampling import Bounds, uniform_points


def _xy(z: np.ndarray) -> np.ndarray:
  return np.column_stack([z.real, z.imag])


def test_triangle_errors_scale_with_area_and_spread() -> None:
  # Two triangles of equal shape; the second is twice as wide in `values`.
  xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
  simplices = np.array([[0, 1, 2], [1, 3, 2]])
  flat = triangle_errors(xy, np.array([0.0, 0.0, 0.0, 0.0]), simplices)
  varying = triangle_errors(xy, np.array([0.0, 0.0, 0.0, 2.0]), simplices)

  assert np.allclose(flat, 0.0)
  # Only the triangle touching the varying vertex picks up error.
  assert varying[1] > varying[0]


def test_triangle_errors_reward_large_triangles() -> None:
  small = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
  large = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
  values = np.array([0.0, 1.0, 0.0])
  simplices = np.array([[0, 1, 2]])

  assert (triangle_errors(large, values, simplices)
          > triangle_errors(small, values, simplices))


def _peaked_field(z: np.ndarray) -> np.ndarray:
  """A field with a sharp ridge, so refinement has something to chase."""
  return 1e-6 + np.abs(z - (0.2 + 0.1j))


def test_refine_respects_the_budget() -> None:
  points = uniform_points(Bounds(-1.0, 1.0, -1.0, 1.0), 6, 6)
  sigmin = _peaked_field(points)
  calls: list[int] = []

  def sample(batch: np.ndarray) -> np.ndarray:
    calls.append(batch.size)
    return _peaked_field(batch)

  out_z, out_s = refine(points, sigmin, sample, budget=100, rounds=4)

  assert out_z.size <= 100
  assert out_z.size > points.size
  assert out_z.size == out_s.size
  assert len(calls) <= 4, 'one batched evaluation per round keeps the pool busy'


@pytest.mark.parametrize(
  ('wanted', 'rounds'),
  [(600, 4), (600, 7), (10, 4), (50, 3), (17, 5)],
)
def test_refine_spends_the_whole_request_when_geometry_allows(
  wanted: int, rounds: int
) -> None:
  """The per-round split must not truncate the remainder away.

  Dividing the request by the round count and flooring used to lose up to
  rounds-1 points: 600 over 7 rounds bought only 595.
  """
  points = uniform_points(Bounds(-1.0, 1.0, -1.0, 1.0), 12, 12)
  sigmin = _peaked_field(points)

  out_z, _ = refine(
    points, sigmin, _peaked_field, budget=points.size + wanted, rounds=rounds)
  assert out_z.size - points.size == wanted


@pytest.mark.parametrize(
  ('nx', 'ny', 'wanted', 'rounds'),
  [(12, 12, 600, 4), (12, 12, 1000, 4), (12, 12, 600, 1), (3, 3, 500, 4)],
)
def test_budget_is_a_ceiling_never_exceeded(
  nx: int, ny: int, wanted: int, rounds: int
) -> None:
  """Even when geometry limits the run, the budget is an upper bound."""
  points = uniform_points(Bounds(-1.0, 1.0, -1.0, 1.0), nx, ny)
  sigmin = _peaked_field(points)

  out_z, out_s = refine(
    points, sigmin, _peaked_field, budget=points.size + wanted, rounds=rounds)
  assert out_z.size <= points.size + wanted
  assert out_s.size == out_z.size


def test_one_round_is_limited_by_the_triangle_count(capsys) -> None:
  """A round inserts at most one point per triangle, so it cannot triple a set.

  This is the structural reason --refine-points is a ceiling, and it must be
  reported rather than silently absorbed.
  """
  points = uniform_points(Bounds(-1.0, 1.0, -1.0, 1.0), 12, 12)
  sigmin = _peaked_field(points)

  out_z, _ = refine(
    points, sigmin, _peaked_field, budget=points.size + 600, rounds=1)
  added = out_z.size - points.size

  assert added < 600, 'a single round should run out of triangles'
  assert added > 2 * points.size * 0.5, 'but it should still roughly triple the set'
  assert 'short because the triangulation ran out' in capsys.readouterr().out


def test_refine_is_a_noop_when_disabled_or_already_full() -> None:
  points = uniform_points(Bounds(-1.0, 1.0, -1.0, 1.0), 5, 5)
  sigmin = _peaked_field(points)

  def sample(batch: np.ndarray) -> np.ndarray:
    raise AssertionError('should not sample')

  assert refine(points, sigmin, sample, budget=100, rounds=0)[0].size == points.size
  assert refine(points, sigmin, sample, budget=10, rounds=3)[0].size == points.size


def test_refine_concentrates_points_near_the_ridge() -> None:
  """Refinement should add points where the field varies, not uniformly."""
  bounds = Bounds(-1.0, 1.0, -1.0, 1.0)
  points = uniform_points(bounds, 8, 8)
  sigmin = _peaked_field(points)
  added = refine(points, sigmin, _peaked_field, budget=200, rounds=4)[0][points.size:]

  # The ridge sits at 0.2+0.1j; added points should cluster nearer it than a
  # uniform draw over the same box would.
  near = np.abs(added - (0.2 + 0.1j)).mean()
  uniform_reference = np.abs(uniform_points(bounds, 20, 20) - (0.2 + 0.1j)).mean()
  assert near < uniform_reference


def test_refine_survives_collinear_seeds() -> None:
  # Collinear points have no valid triangulation; refinement must give up
  # rather than abort a long job.
  points = np.linspace(-1, 1, 10).astype(np.complex128)
  sigmin = np.ones(10)
  out_z, out_s = refine(points, sigmin, _peaked_field, budget=50, rounds=3)
  assert out_z.size == points.size
  assert out_s.size == sigmin.size


def test_refine_survives_too_few_points() -> None:
  points = np.array([0 + 0j, 1 + 1j], dtype=np.complex128)
  sigmin = np.array([1.0, 2.0])
  assert refine(points, sigmin, _peaked_field, budget=50, rounds=3)[0].size == 2


def _grcar(n: int, k: int = 3) -> np.ndarray:
  """The Grcar matrix: a classic strongly non-normal pseudospectra test case."""
  G = np.zeros((n, n))
  for i in range(n):
    if i + 1 < n:
      G[i + 1, i] = -1.0
    for j in range(k + 1):
      if i + j < n:
        G[i, i + j] = 1.0
  return G


def test_adaptive_beats_uniform_at_equal_budget() -> None:
  """Refinement must earn its complexity: lower error for the same point count.

  Interpolation error is measured in log10 against a dense-SVD reference, which
  is ground truth. If this regresses, adaptive refinement is not paying for
  itself and should be reconsidered -- upstream deleted an earlier refinement
  scheme for exactly that reason.

  Grcar(60) is used because its dense-SVD reference is cheap. The same
  comparison on west0479 (order 479, normality defect 0.63) measured 2.05x
  during development.
  """
  n = 60
  A = _grcar(n)
  eigvals = np.linalg.eigvals(A)
  T = np.asarray(scipy.linalg.schur(A, output='complex')[0], dtype=np.complex128)
  bounds = Bounds.around_spectrum(eigvals, pad=0.25)

  # Reference mesh, offset half a cell so no sample can land on a reference
  # point and score a trivially perfect interpolation.
  side = 19
  dx = (bounds.real_max - bounds.real_min) / (2 * side)
  dy = (bounds.imag_max - bounds.imag_min) / (2 * side)
  xr = np.linspace(bounds.real_min + dx, bounds.real_max - dx, side)
  yr = np.linspace(bounds.imag_min + dy, bounds.imag_max - dy, side)
  Xr, Yr = np.meshgrid(xr, yr)
  Zref = Xr + 1j * Yr
  eye = np.eye(n)
  truth = np.log10(np.array(
    [[np.linalg.svd(z * eye - T, compute_uv=False)[-1] for z in row] for row in Zref]))

  def interp_error(z: np.ndarray, s: np.ndarray) -> float:
    got = LinearNDInterpolator(
      np.column_stack([z.real, z.imag]), np.log10(s))(Zref.real, Zref.imag)
    covered = np.isfinite(got)
    assert covered.mean() > 0.99, 'sample hull should cover the reference mesh'
    return float(np.abs(got[covered] - truth[covered]).mean())

  # Equal budget: a 20x20 uniform lattice, versus a coarse 12x12 start plus the
  # same number of extra evaluations spent adaptively.
  budget = 400

  z_uniform = uniform_points(bounds, 20, 20)
  err_uniform = interp_error(z_uniform, sample_sigmin(z_uniform, T, 2))

  z_seed = uniform_points(bounds, 12, 12)
  z_adaptive, s_adaptive = refine(
    z_seed, sample_sigmin(z_seed, T, 2),
    lambda batch: sample_sigmin(batch, T, 2), budget=budget, rounds=4)
  err_adaptive = interp_error(z_adaptive, s_adaptive)

  assert z_adaptive.size <= z_uniform.size, 'adaptive must not overspend the budget'
  assert err_adaptive < err_uniform, (
    f'adaptive {err_adaptive:.4f} did not beat uniform {err_uniform:.4f}')


def test_triangulation_of_refined_set_is_valid() -> None:
  points = uniform_points(Bounds(-1.0, 1.0, -1.0, 1.0), 6, 6)
  sigmin = _peaked_field(points)
  out_z, _ = refine(points, sigmin, _peaked_field, budget=150, rounds=3)

  # No duplicate points, and Qhull accepts the result.
  assert np.unique(out_z).size == out_z.size
  assert Delaunay(_xy(out_z)).simplices.size > 0
