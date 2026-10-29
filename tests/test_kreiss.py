"""The Kreiss bound: a real-axis search, checked against theory and dense references."""

import math

import numpy as np
import pytest
import scipy.linalg

from nonmodal.kreiss import kreiss_constant


def _schur(a: np.ndarray) -> np.ndarray:
  return np.asarray(scipy.linalg.schur(a, output='complex')[0], dtype=np.complex128)


def _shifted_jordan(n: int = 30, coupling: float = 1.5) -> np.ndarray:
  """Stable (all eigenvalues -1) and strongly non-normal."""
  return -np.eye(n) + coupling * np.eye(n, k=1)


def _complex_jordan(a: float = 1.0, b: float = 50.0, c: float = 40.0) -> np.ndarray:
  """A real operator whose non-normality sits at a complex pair -a +- ib, off the axis."""
  rot = np.array([[-a, b], [-b, -a]])
  zero, eye = np.zeros((2, 2)), c * np.eye(2)
  return np.block([[rot, eye, zero], [zero, rot, eye], [zero, zero, rot]])


def test_a_normal_operator_approaches_one_from_below() -> None:
  """K = 1 for a stable normal operator, approached only as Re z -> infinity."""
  result = kreiss_constant(_schur(np.diag(-np.arange(1.0, 11.0))), nprocs=2)
  assert 0.95 < result.kreiss <= 1.0
  assert result.axis_at_upper_end
  assert not result.offaxis_exceeds_axis


def test_the_kreiss_matrix_theorem_holds() -> None:
  """K <= sup_t ||e^{tA}|| <= e n K, so a wrong K fails rather than looking odd."""
  a = _shifted_jordan()
  result = kreiss_constant(_schur(a), nprocs=4)
  growth = max(np.linalg.norm(scipy.linalg.expm(t * a), 2) for t in np.linspace(0.0, 60.0, 601))

  assert result.kreiss <= growth * (1 + 1e-9)
  assert growth <= math.e * a.shape[0] * result.kreiss
  assert result.upper_bound == pytest.approx(math.e * a.shape[0] * result.kreiss)


def test_the_real_axis_maximum_matches_a_dense_reference() -> None:
  a = _shifted_jordan()
  result = kreiss_constant(_schur(a), nprocs=4)
  xs = np.geomspace(1e-3, 1e3, 3000)
  dense = max(x / np.linalg.svd(x * np.eye(a.shape[0]) - a, compute_uv=False)[-1] for x in xs)

  assert result.axis_kreiss == pytest.approx(dense, rel=1e-5)
  assert result.axis_is_local_max_in_y
  assert not result.offaxis_exceeds_axis
  assert result.z_star_imag == 0.0


def test_an_off_axis_supremum_is_found_and_flagged() -> None:
  """Where the real axis is a saddle, the check must say so and report the larger value."""
  a = _complex_jordan()
  result = kreiss_constant(_schur(a), nprocs=4)

  assert result.offaxis_exceeds_axis
  assert not result.axis_is_local_max_in_y
  assert result.kreiss == result.offaxis_kreiss > 2 * result.axis_kreiss
  # Every reported value is a real evaluation, so a lower bound on the true K.
  z = complex(result.z_star_real, result.z_star_imag)
  sigma = np.linalg.svd(z * np.eye(a.shape[0]) - a, compute_uv=False)[-1]
  assert result.kreiss == pytest.approx(z.real / sigma, rel=1e-6)
  growth = max(np.linalg.norm(scipy.linalg.expm(t * a), 2) for t in np.linspace(0.0, 10.0, 2001))
  assert result.kreiss <= growth * (1 + 1e-9)


def test_an_unstable_operator_is_rejected() -> None:
  with pytest.raises(ValueError, match='not stable'):
    kreiss_constant(_schur(np.diag([-1.0, 0.5])), nprocs=1)
