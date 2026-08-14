"""Eigenvalues and eigenvectors taken from the Schur factor rather than a solver."""

import numpy as np
import pytest
import scipy

from nonmodal.operator import (
  eigenvectors_from_schur,
  rightmost_indices,
  spectrum_from_schur,
)


def _factorised(n: int = 24, seed: int = 7):
  """A real operator with its complex Schur factorisation `A = Z T Z*`."""
  rng = np.random.default_rng(seed)
  a = rng.standard_normal((n, n))
  schur_t, schur_z = scipy.linalg.schur(a, output='complex')
  return a, np.asarray(schur_t), np.asarray(schur_z)


def test_spectrum_is_the_schur_diagonal(tmp_path) -> None:
  a, schur_t, _ = _factorised()
  eigvals = spectrum_from_schur(schur_t, str(tmp_path))

  # Matched by distance, since a conjugate pair's sort order is not stable.
  reference = np.linalg.eigvals(a)
  assert eigvals.size == reference.size
  assert np.abs(eigvals[:, None] - reference[None, :]).min(axis=1).max() < 1e-9
  # Cached for consumers that want the spectrum without the factor.
  cached = np.load(tmp_path / 'full_reduced_eigvals.npy')
  assert np.array_equal(cached, eigvals)


def test_eigenvectors_satisfy_the_eigenvalue_equation() -> None:
  """The defining property, checked on A itself rather than on T."""
  a, schur_t, schur_z = _factorised()
  eigvals = schur_t.diagonal()
  indices = np.arange(eigvals.size, dtype=np.intp)

  vecs = eigenvectors_from_schur(schur_t, schur_z, indices)

  for col, lam in enumerate(eigvals):
    v = vecs[:, col]
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert np.linalg.norm(a @ v - lam * v) < 1e-8


def test_eigenvectors_match_a_dense_solver() -> None:
  """Same invariant subspaces as `scipy.linalg.eig`, up to phase."""
  a, schur_t, schur_z = _factorised()
  ref_vals, ref_vecs = scipy.linalg.eig(a)

  indices = rightmost_indices(schur_t.diagonal(), 4)
  vecs = eigenvectors_from_schur(schur_t, schur_z, indices)

  for col, i in enumerate(indices):
    j = int(np.abs(ref_vals - schur_t[i, i]).argmin())
    ref = ref_vecs[:, j] / np.linalg.norm(ref_vecs[:, j])
    # Eigenvectors are defined up to a phase, so compare the alignment.
    assert abs(np.vdot(ref, vecs[:, col])) == pytest.approx(1.0, abs=1e-7)


def test_selection_is_by_real_part_alone() -> None:
  """Selection is by real part alone: both members of a pair are kept.

  A roundoff imaginary part of ~5e-9 must not disqualify a real eigenvalue.
  """
  eigvals = np.array([-1 + 2j, -1 - 2j, -5 + 0j, -3 + 1j, -3 - 1j])

  kept = rightmost_indices(eigvals, 4)
  assert [complex(z) for z in eigvals[kept]] == [-1 + 2j, -1 - 2j, -3 + 1j, -3 - 1j]

  # A roundoff imaginary part must not disqualify an otherwise real eigenvalue.
  noisy = np.array([-27.86 + 5.8e-9j, -27.90 - 5.8e-9j, -701.5 + 1.0e-8j])
  assert rightmost_indices(noisy, 3).size == 3


def test_rightmost_indices_are_ordered_by_real_part() -> None:
  _, schur_t, _ = _factorised()
  chosen = schur_t.diagonal()[rightmost_indices(schur_t.diagonal(), 5)]
  assert np.all(np.diff(chosen.real) <= 0.0)


def test_near_degenerate_diagonal_does_not_blow_up() -> None:
  """A repeated eigenvalue clamps rather than dividing by ~0."""
  n = 6
  schur_t = np.triu(np.ones((n, n), dtype=np.complex128))
  # Two identical diagonal entries, so back-substitution meets a zero denominator.
  schur_t[np.diag_indices(n)] = np.array([2.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  schur_z = np.eye(n, dtype=np.complex128)

  vecs = eigenvectors_from_schur(schur_t, schur_z, np.array([1], dtype=np.intp))
  assert np.all(np.isfinite(vecs))
  assert np.linalg.norm(vecs[:, 0]) == pytest.approx(1.0)


def test_shape_mismatch_is_rejected() -> None:
  _, schur_t, schur_z = _factorised()
  with pytest.raises(ValueError, match='disagree'):
    eigenvectors_from_schur(schur_t, schur_z[:, :-1], np.array([0], dtype=np.intp))
