"""Spectral truncation onto an invariant subspace: exactness and selection."""

import numpy as np
import pytest
import scipy.linalg

from nonmodal.operator import eigenvectors_from_schur
from nonmodal.pseudomode import pseudomode_at
from nonmodal.resolution import ModeResolution
from nonmodal.subspace import (
  Subspace,
  conjugate_closed,
  restrict_schur,
  restrict_to_resolved,
  whole_operator,
)


def _schur(a):
  t, z = scipy.linalg.schur(a.astype(np.complex128), output='complex')
  return t, z


def _sigmin(m):
  return float(scipy.linalg.svdvals(m)[-1])


def test_restriction_is_an_exact_invariant_subspace():
  rng = np.random.default_rng(1)
  a = rng.standard_normal((30, 30))
  t, z = _schur(a)
  keep = np.abs(np.diag(t)) < 3.0
  t11, z1 = restrict_schur(t, z, keep)

  assert t11.shape == (keep.sum(), keep.sum())
  assert np.allclose(np.tril(t11, -1), 0.0)
  assert np.allclose(z1.conj().T @ z1, np.eye(keep.sum()), atol=1e-12)
  # A Z1 = Z1 T11: the restriction is exact, not a projection.
  assert np.abs(a @ z1 - z1 @ t11).max() < 1e-12
  kept = np.sort_complex(np.diag(t11))
  assert np.allclose(kept, np.sort_complex(np.diag(t)[keep]), atol=1e-12)


def test_restriction_without_vectors_matches():
  rng = np.random.default_rng(2)
  t, z = _schur(rng.standard_normal((20, 20)))
  keep = np.diag(t).real > -1.0
  before = t.copy()
  with_z, _ = restrict_schur(t, z, keep)
  without_z, none = restrict_schur(t, None, keep)
  assert np.array_equal(t, before), 'the caller\'s T must not be overwritten'
  assert none is None
  # T11 is the same up to rounding in the reordering.
  assert np.allclose(with_z, without_z, atol=1e-12)


def test_restriction_never_raises_the_resolvent_norm():
  # (zI - B)^-1 restricted to an invariant subspace is (zI - T11)^-1 there.
  rng = np.random.default_rng(3)
  a = rng.standard_normal((25, 25)) + 4.0 * np.triu(rng.standard_normal((25, 25)), 1)
  t, _ = _schur(a)
  keep = np.abs(np.diag(t)) < np.median(np.abs(np.diag(t)))
  t11, _ = restrict_schur(t, None, keep)
  for zz in (0.5 + 0.5j, 3.0, -2.0 + 4.0j, 10.0 + 1.0j):
    full = _sigmin(zz * np.eye(25) - t)
    sub = _sigmin(zz * np.eye(t11.shape[0]) - t11)
    assert sub >= full * (1.0 - 1e-10)


def test_restriction_strips_a_decoupled_non_normal_grid_block():
  # Slow normal block plus a fast, strongly non-normal one: what the truncation
  # is for. Restricted, the pseudospectrum is the slow block's alone.
  slow = np.diag([-1.0, -2.0, -3.0])
  fast = -1e3 * np.eye(4) + 1e5 * np.diag(np.ones(3), 1)
  a = scipy.linalg.block_diag(slow, fast)
  q, _ = np.linalg.qr(np.random.default_rng(4).standard_normal((7, 7)))
  t, _ = _schur(q @ a @ q.T)  # an orthogonal change of basis keeps the norm
  t11, _ = restrict_schur(t, None, np.abs(np.diag(t)) < 10.0)

  z = 50.0
  full = _sigmin(z * np.eye(7) - t)
  sub = _sigmin(z * np.eye(3) - t11)
  # The slow block is normal: sigma_min is the distance to its spectrum.
  assert sub == pytest.approx(z + 1.0, rel=1e-10)
  # The fast block's non-normality dominates the unrestricted resolvent.
  assert full < 1e-2 * sub


def test_restriction_rejects_bad_input():
  t, z = _schur(np.random.default_rng(5).standard_normal((6, 6)))
  with pytest.raises(ValueError, match='keeps no'):
    restrict_schur(t, z, np.zeros(6, dtype=bool))
  with pytest.raises(ValueError, match='entries'):
    restrict_schur(t, z, np.ones(5, dtype=bool))
  with pytest.raises(ValueError, match='do not match'):
    restrict_schur(t, z[:, :4], np.ones(6, dtype=bool))


def test_subspace_record_is_json_ready():
  import json

  record = Subspace(restricted=True, dimension=2, full_dimension=5, rule='per-field',
                    cutoff=0.1, k_max={'U_n': 38.2}, max_abs_eigenvalue=3.0,
                    kept_eigenvalues=(1 + 2j, -3.0 + 0j))
  data = json.loads(json.dumps(record.describe()))
  assert data['kept_eigenvalues'] == [[1.0, 2.0], [-3.0, 0.0]]
  assert data['restricted'] is True


# --- restriction end to end -------------------------------------------------


def _resolution(t, worst):
  """A resolution record whose single field has `k / k_max = worst`, per mode."""
  worst = np.asarray(worst, dtype=float)
  return ModeResolution(
    eigenvalues=np.diag(t).copy(), fields=(0,), wavenumber=worst[:, None] * 10.0,
    share=np.ones((worst.size, 1)), k_max=np.array([10.0]))


def test_whole_operator_record() -> None:
  t, _ = _schur(np.random.default_rng(6).standard_normal((5, 5)))
  record = whole_operator(t)
  assert not record.restricted and record.dimension == record.full_dimension == 5


def test_restriction_keeps_the_resolved_modes() -> None:
  rng = np.random.default_rng(7)
  q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
  slow = np.array([[-1.0, 2.0], [-2.0, -1.0]])  # -1 +- 2i
  a = q @ scipy.linalg.block_diag(slow, -3.0, -1e4, -2e4, -3e4) @ q.T
  t, z = _schur(a)
  # Resolution grows with |lambda|: the three slow modes are smooth.
  worst = np.where(np.abs(np.diag(t)) < 10.0, 0.05, 0.5)

  t11, z1, record = restrict_to_resolved(t, z, _resolution(t, worst), 0.1, 'k.h5')
  assert record.restricted and record.dimension == 3 and record.full_dimension == 6
  assert record.conjugate_closed and record.cutoff == pytest.approx(0.1)
  assert record.k_max == {'U_n': 10.0}
  assert record.max_abs_eigenvalue == pytest.approx(3.0)
  assert np.abs(a @ z1 - z1 @ t11).max() < 1e-11


def test_a_resolution_for_another_factor_is_rejected() -> None:
  t, z = _schur(np.diag([-1.0, -2.0]))
  stale = _resolution(np.diag([-1.0, -2.5]), [0.0, 0.0])
  with pytest.raises(ValueError, match='different Schur factor'):
    restrict_to_resolved(t, z, stale, 0.1, 'k.h5')


def test_no_resolved_mode_raises() -> None:
  t, z = _schur(np.diag([-1.0, -2.0]))
  with pytest.raises(ValueError, match='no mode is resolved'):
    restrict_to_resolved(t, z, _resolution(t, [0.3, 0.4]), 0.1, 'k.h5')


def test_eigenvectors_and_pseudomodes_work_on_a_tall_z() -> None:
  rng = np.random.default_rng(8)
  a = rng.standard_normal((12, 12)) + 3.0 * np.triu(rng.standard_normal((12, 12)), 1)
  t, z = _schur(a)
  keep = np.abs(np.diag(t)) < np.median(np.abs(np.diag(t)))
  t11, z1 = restrict_schur(t, z, keep)

  vecs = eigenvectors_from_schur(t11, z1, np.arange(t11.shape[0]))
  for k in range(t11.shape[0]):
    assert np.linalg.norm(a @ vecs[:, k] - t11[k, k] * vecs[:, k]) < 1e-9

  mode = pseudomode_at(t11, z1, 0.3 + 0.7j)
  x = mode.vector
  # A pseudomode of the restriction is a pseudomode of A that lives in the subspace.
  assert np.linalg.norm(a @ x - (0.3 + 0.7j) * x) == pytest.approx(mode.sigma_min, rel=1e-7)
  assert np.linalg.norm(x - z1 @ (z1.conj().T @ x)) < 1e-10


def test_conjugate_closure() -> None:
  assert conjugate_closed(np.array([-1 + 2j, -1 - 2j, -3 + 5e-9j]))
  assert not conjugate_closed(np.array([-1 + 2j, -3.0]))
  assert conjugate_closed(np.array([0.0 + 0j]))
