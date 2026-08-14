"""Pseudospectrum sampling and contour level selection.

The sigma_min tests are the load-bearing ones: for a diagonal (hence already
Schur-form) operator, sigma_min(zI - T) = min_i |z - T_ii| in closed form, so
the LAPACK triangular-solve / eigsh inverse-iteration path can be checked
exactly rather than against a golden file.
"""

import numpy as np
import pytest
import scipy.linalg

from nonmodal.pseudospectrum import (
  _worker_count,
  choose_contour_levels,
  compute_pseudospectrum,
  sample_sigmin,
)
from nonmodal.sampling import Bounds, mirror_conjugates


def _exact_sigmin(
  R: np.ndarray, C: np.ndarray, diag: np.ndarray
) -> np.ndarray:
  """Closed-form sigma_min for a diagonal operator."""
  Z = R + 1j * C
  return np.min(np.abs(Z[..., None] - diag[None, None, :]), axis=-1)


def test_sigmin_matches_closed_form_on_a_centred_grid() -> None:
  diag = np.array([1 + 0.5j, 1 - 0.5j, -2 + 1j, -2 - 1j], dtype=np.complex128)
  R, C, sigmin = compute_pseudospectrum(
    np.diag(diag), Bounds(-4.0, 4.0, -3.0, 3.0), nx=4, ny=4, nprocs=1)

  np.testing.assert_allclose(sigmin, _exact_sigmin(R, C, diag), rtol=1e-8)


def test_sigmin_matches_closed_form_off_centre() -> None:
  # compute_pseudospectrum never mirrors, so the spectrum need not be symmetric.
  diag = np.array([1 + 0.5j, -2 + 1j, 0.5 - 1.5j, 3 + 0j], dtype=np.complex128)
  R, C, sigmin = compute_pseudospectrum(
    np.diag(diag), Bounds(-4.0, 4.0, -3.0, 2.5), nx=4, ny=4, nprocs=1)

  np.testing.assert_allclose(sigmin, _exact_sigmin(R, C, diag), rtol=1e-8)


def test_bounds_reject_inverted_regions() -> None:
  with pytest.raises(ValueError, match='strictly ordered'):
    Bounds(1.0, -1.0, -1.0, 1.0)


@pytest.mark.parametrize(
  ('total', 'requested', 'expected'),
  [(100, 8, 8), (4, 8, 4), (100, 0, 1), (100, -5, 1)],
)
def test_worker_count_clamps(total: int, requested: int, expected: int) -> None:
  assert _worker_count(total, requested) == expected


def test_sample_sigmin_handles_empty_input() -> None:
  T = np.diag([1 + 0j, 2 + 0j]).astype(np.complex128)
  assert sample_sigmin(np.zeros(0, dtype=np.complex128), T, 1).size == 0


def test_sampling_is_bitwise_reproducible() -> None:
  """Repeating a run gives bitwise identical numbers, via `_start_vector`."""
  rng = np.random.default_rng(11)
  T = np.triu(rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10)))
  z = np.array([0.3 + 0.4j, -1.0 + 0.2j, 2.0 - 0.5j], dtype=np.complex128)

  first = sample_sigmin(z, T.astype(np.complex128), 1)
  second = sample_sigmin(z, T.astype(np.complex128), 1)
  assert np.array_equal(first, second)


def test_half_plane_mirroring_matches_full_sampling() -> None:
  """A real operator's pseudospectrum is symmetric, so mirroring is exact.

  Sampling the upper half-plane and conjugating reproduces full sampling.
  """
  rng = np.random.default_rng(3)
  A = rng.normal(size=(8, 8))
  assert np.isrealobj(A)
  T = np.asarray(scipy.linalg.schur(A, output='complex')[0], dtype=np.complex128)

  z = np.array([0.4 + 0.9j, -1.2 + 0.3j, 0.7 + 0.0j], dtype=np.complex128)
  upper = sample_sigmin(z, T, 1)
  mirrored_z, mirrored_s = mirror_conjugates(z, upper)

  direct = sample_sigmin(np.conj(z[z.imag != 0]), T, 1)
  np.testing.assert_allclose(mirrored_s[z.size:], direct, rtol=1e-6)
  np.testing.assert_allclose(mirrored_z[z.size:], np.conj(z[z.imag != 0]))


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
  # A constant field still yields two distinct boundaries.
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
