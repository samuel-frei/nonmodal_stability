"""Pseudospectrum sampling and contour level selection.

The sigma_min tests are the load-bearing ones: for a diagonal (hence already
Schur-form) operator, sigma_min(zI - T) = min_i |z - T_ii| in closed form, so
the LAPACK triangular-solve / eigsh inverse-iteration path can be checked
exactly rather than against a golden file.
"""

import numpy as np
import pytest

from nonmodal.pseudospectrum import (
  _worker_count,
  choose_contour_levels,
  compute_pseudospectrum,
)


def _exact_sigmin(
  R: np.ndarray, C: np.ndarray, diag: np.ndarray
) -> np.ndarray:
  """Closed-form sigma_min for a diagonal operator."""
  Z = R + 1j * C
  return np.min(np.abs(Z[..., None] - diag[None, None, :]), axis=-1)


def test_sigmin_matches_closed_form_on_mirrored_grid() -> None:
  # Conjugate-symmetric spectrum, so mirroring the upper half-plane is valid.
  diag = np.array([1 + 0.5j, 1 - 0.5j, -2 + 1j, -2 - 1j], dtype=np.complex128)
  R, C, sigmin = compute_pseudospectrum(
    np.diag(diag), grid_points=16, nprocs=1,
    real_min=-4.0, real_max=4.0, imag_min=-3.0, imag_max=3.0)

  np.testing.assert_allclose(sigmin, _exact_sigmin(R, C, diag), rtol=1e-8)


def test_sigmin_matches_closed_form_on_full_grid() -> None:
  # Asymmetric imaginary bounds force full-grid evaluation, so an arbitrary
  # (non-conjugate-symmetric) spectrum is fair game here.
  diag = np.array([1 + 0.5j, -2 + 1j, 0.5 - 1.5j, 3 + 0j], dtype=np.complex128)
  R, C, sigmin = compute_pseudospectrum(
    np.diag(diag), grid_points=16, nprocs=1,
    real_min=-4.0, real_max=4.0, imag_min=-3.0, imag_max=2.5)

  np.testing.assert_allclose(sigmin, _exact_sigmin(R, C, diag), rtol=1e-8)


def test_compute_pseudospectrum_requires_bounds() -> None:
  T = np.diag([1 + 0j, 2 + 0j]).astype(np.complex128)
  with pytest.raises(ValueError, match='bounds must all be provided'):
    compute_pseudospectrum(T, grid_points=4, nprocs=1, real_min=-1.0, real_max=1.0)


def test_compute_pseudospectrum_rejects_inverted_bounds() -> None:
  T = np.diag([1 + 0j, 2 + 0j]).astype(np.complex128)
  with pytest.raises(ValueError, match='real_max must be greater'):
    compute_pseudospectrum(
      T, grid_points=4, nprocs=1,
      real_min=1.0, real_max=-1.0, imag_min=-1.0, imag_max=1.0)


@pytest.mark.parametrize(
  ('total', 'requested', 'expected'),
  [(100, 8, 8), (4, 8, 4), (100, 0, 1), (100, -5, 1)],
)
def test_worker_count_clamps(total: int, requested: int, expected: int) -> None:
  assert _worker_count(total, requested) == expected


def test_contour_levels_are_geometric_and_bounded() -> None:
  sigmin = np.array([[1e-6, 1e-4], [1e-2, 1.0]])
  levels = choose_contour_levels(sigmin, min_level=1e-6, nlevels=4)

  assert levels.shape == (4,)
  assert levels[0] == pytest.approx(1e-6)
  assert levels[-1] == pytest.approx(1.0)
  ratios = levels[1:] / levels[:-1]
  np.testing.assert_allclose(ratios, ratios[0], rtol=1e-9)


def test_contour_levels_clamp_min_level_to_data() -> None:
  sigmin = np.array([[1e-3, 1e-2]])
  levels = choose_contour_levels(sigmin, min_level=1e-9, nlevels=3)
  assert levels[0] == pytest.approx(1e-3)


def test_contour_levels_near_constant_field() -> None:
  # Degenerate field: two distinct boundaries are still returned so contouring
  # has something to draw.
  levels = choose_contour_levels(np.full((3, 3), 0.25), min_level=1e-7, nlevels=5)
  assert levels.shape == (2,)
  assert levels[1] > levels[0]


def test_contour_levels_reject_bad_input() -> None:
  with pytest.raises(ValueError, match='nlevels must be >= 1'):
    choose_contour_levels(np.ones((2, 2)), nlevels=0)
  with pytest.raises(ValueError, match='min_level must be positive'):
    choose_contour_levels(np.ones((2, 2)), min_level=0.0)
  with pytest.raises(ValueError, match='no finite positive entries'):
    choose_contour_levels(np.zeros((2, 2)))
