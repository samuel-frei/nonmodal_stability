"""Pseudomode extraction and restart output.

The load-bearing checks are reference-based: a pseudomode is validated against a
dense SVD of `zI - A`, and against the identity `||(A - zI)v|| = sigma_min` that
defines it. Neither shares code with the inverse iteration under test.
"""

import numpy as np
import pytest
import scipy.linalg

from nonmodal.fields import FIELD_NAMES, aligned_phase, write_restart_modes
from nonmodal.pseudomode import (
  SAMPLE_RULES,
  pseudomode_at,
  select_from_samples,
)

h5py = pytest.importorskip('h5py')


def _schur(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  t, z = scipy.linalg.schur(a, output='complex')
  return np.asarray(t, dtype=np.complex128), np.asarray(z, dtype=np.complex128)


def _non_normal(n: int = 40) -> np.ndarray:
  """A real, decidedly non-normal matrix."""
  rng = np.random.default_rng(5)
  return np.asarray(rng.normal(size=(n, n)))


# --- the mode itself -------------------------------------------------------


def test_pseudomode_of_a_diagonal_operator_is_a_basis_vector() -> None:
  """For a normal operator sigma_min = dist(z, spectrum), attained on one axis."""
  diag = np.array([1 + 0.5j, -2 + 1j, 3 - 1j, 0.5 + 2j], dtype=np.complex128)
  a = np.diag(diag)
  z = 1.05 + 0.45j  # nearest to diag[0]
  t, zv = _schur(a)

  mode = pseudomode_at(t, zv, z)

  assert mode.sigma_min == pytest.approx(np.min(np.abs(z - diag)), rel=1e-8)
  # All the weight sits on the nearest eigenvalue's axis.
  weights = np.abs(mode.vector)
  assert int(np.argmax(weights)) == int(np.argmin(np.abs(z - diag)))
  assert weights.max() == pytest.approx(1.0, rel=1e-6)


def test_pseudomode_matches_a_dense_svd() -> None:
  a = _non_normal()
  z = 0.4 - 0.9j
  t, zv = _schur(a)

  mode = pseudomode_at(t, zv, z)

  m = z * np.eye(a.shape[0]) - a
  u, s, vh = np.linalg.svd(m)
  assert mode.sigma_min == pytest.approx(float(s[-1]), rel=1e-9)
  # Singular vectors are defined up to phase, so compare the overlap.
  assert abs(np.vdot(mode.vector, vh[-1].conj())) == pytest.approx(1.0, rel=1e-8)
  assert np.linalg.norm(mode.vector) == pytest.approx(1.0, rel=1e-12)


def test_residual_identifies_the_pseudomode() -> None:
  """||(A - zI)v|| == sigma_min is what makes v a pseudomode rather than noise."""
  a = _non_normal()
  z = -0.7 + 1.3j
  t, zv = _schur(a)

  mode = pseudomode_at(t, zv, z)

  # Reported residual, and the same quantity recomputed in the physical basis.
  direct = float(np.linalg.norm(a @ mode.vector - z * mode.vector))
  assert mode.residual == pytest.approx(mode.sigma_min, rel=1e-8)
  assert direct == pytest.approx(mode.sigma_min, rel=1e-8)


def test_pseudomode_beats_the_nearest_eigenvector_on_a_non_normal_operator() -> None:
  """The point of the exercise: sigma_min is far below the eigenvalue distance."""
  a = _non_normal()
  eigvals = np.linalg.eigvals(a)
  z = 2.5 + 2.5j  # deliberately far from the spectrum
  t, zv = _schur(a)

  mode = pseudomode_at(t, zv, z)

  assert mode.sigma_min < np.min(np.abs(z - eigvals))


def test_pseudomode_rejects_mismatched_factors() -> None:
  t, zv = _schur(_non_normal(12))
  with pytest.raises(ValueError, match='disagree'):
    pseudomode_at(t, zv[:, :5], 0.1 + 0.1j)


# --- choosing a point out of a finished run --------------------------------


def _samples() -> tuple[np.ndarray, np.ndarray]:
  z = np.array([1 + 0j, 2 + 1j, -1 - 1j, 2 - 1j, 0.5 + 0.5j], dtype=np.complex128)
  sigmin = np.array([0.5, 0.1, 0.01, 0.4, 0.2])
  return z, sigmin


@pytest.mark.parametrize('rule', SAMPLE_RULES)
def test_every_rule_returns_one_of_the_samples(rule: str) -> None:
  z, sigmin = _samples()
  point, _ = select_from_samples(z, sigmin, rule)
  assert point in set(z.tolist())


def test_rules_pick_the_documented_point() -> None:
  z, sigmin = _samples()
  assert select_from_samples(z, sigmin, 'min-sigmin')[0] == complex(-1, -1)
  # Largest Re z, ties broken by smallest sigma_min: 2+1j over 2-1j.
  assert select_from_samples(z, sigmin, 'rightmost')[0] == complex(2, 1)
  # Largest Re z / sigma_min over the right half-plane.
  assert select_from_samples(z, sigmin, 'kreiss')[0] == complex(2, 1)


def test_boundary_flag_marks_a_point_on_the_sampled_edge() -> None:
  """A rightmost sample is the edge of the box, not the operator's extent."""
  z, sigmin = _samples()
  _, on_boundary = select_from_samples(z, sigmin, 'rightmost')
  assert on_boundary is True


def test_boundary_flag_is_false_for_an_interior_point() -> None:
  z = np.array([0 + 0j, 2 + 2j, 2 - 2j, -2 + 2j, -2 - 2j, 0.1 + 0.1j])
  sigmin = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.001])
  point, on_boundary = select_from_samples(z, sigmin, 'min-sigmin')
  assert point == complex(0.1, 0.1)
  assert on_boundary is False


def test_kreiss_rule_needs_the_right_half_plane() -> None:
  z = np.array([-1 + 0j, -2 + 1j], dtype=np.complex128)
  with pytest.raises(ValueError, match='right half-plane'):
    select_from_samples(z, np.array([0.1, 0.2]), 'kreiss')


def test_unknown_rule_is_rejected() -> None:
  z, sigmin = _samples()
  with pytest.raises(ValueError, match='unknown from-samples rule'):
    select_from_samples(z, sigmin, 'leftmost')


def test_empty_sample_set_is_rejected() -> None:
  with pytest.raises(ValueError, match='no samples'):
    select_from_samples(np.zeros(0, dtype=np.complex128), np.zeros(0), 'rightmost')


# --- restart output --------------------------------------------------------


def _keep(n_global: int = 70, n_kept: int = 50) -> np.ndarray:
  keep = np.zeros(n_global, dtype=bool)
  keep[:n_kept] = True
  return keep


def _read_global(path: str) -> np.ndarray:
  with h5py.File(path, 'r') as f:
    return np.concatenate([np.asarray(f[name]) for name in FIELD_NAMES])


def test_aligned_phase_maximises_amplitude() -> None:
  rng = np.random.default_rng(3)
  v = rng.normal(size=50) + 1j * rng.normal(size=50)

  best = np.linalg.norm(np.real(v * np.exp(-1j * aligned_phase(v))))
  brute = max(
    np.linalg.norm(np.real(v * np.exp(-1j * t)))
    for t in np.linspace(0.0, 2 * np.pi, 4000))

  assert best == pytest.approx(brute, rel=1e-6)
  # And it is a real improvement over taking the real part as-is.
  assert best >= np.linalg.norm(v.real)


def test_restart_round_trip_carries_every_field_block(tmp_path) -> None:
  rng = np.random.default_rng(4)
  keep = _keep()
  v = (rng.normal(size=50) + 1j * rng.normal(size=50)).reshape(-1, 1)

  paths = write_restart_modes(v, keep, 7, str(tmp_path), phases=1)

  assert len(paths) == 1
  with h5py.File(paths[0], 'r') as f:
    assert set(FIELD_NAMES) <= set(f.keys())
    assert float(np.asarray(f['t'])[0]) == 0.0
  # The dropped entries stay zero, and the kept ones round-trip.
  values = _read_global(paths[0])
  np.testing.assert_allclose(
    values[keep], np.real(v[:, 0] * np.exp(-1j * aligned_phase(v[:, 0]))))
  assert not values[~keep].any()


def test_phase_sweep_writes_a_sequence(tmp_path) -> None:
  rng = np.random.default_rng(6)
  keep = _keep()
  v = (rng.normal(size=50) + 1j * rng.normal(size=50)).reshape(-1, 1)

  paths = write_restart_modes(v, keep, 7, str(tmp_path), phases=8)

  assert len(paths) == 8
  # `t` is the file index, which is what makes the directory a time series.
  times = [float(np.asarray(h5py.File(p, 'r')['t'])[0]) for p in paths]
  assert times == list(range(8))

  frames = [_read_global(p) for p in paths]
  amplitudes = [float(np.linalg.norm(f)) for f in frames]
  # Frame 0 is the aligned phase, so it carries the most amplitude...
  assert amplitudes[0] == pytest.approx(max(amplitudes), rel=1e-12)
  # ...and the frames are genuinely different from one another.
  assert not np.allclose(frames[0], frames[1])


def test_phases_must_be_positive(tmp_path) -> None:
  v = np.ones((50, 1), dtype=np.complex128)
  with pytest.raises(ValueError, match='phases must be >= 1'):
    write_restart_modes(v, _keep(), 7, str(tmp_path), phases=0)


def test_restart_rejects_wrong_reduced_size(tmp_path) -> None:
  v = np.ones((7, 1), dtype=np.complex128)
  with pytest.raises(ValueError, match='do not match reduced size'):
    write_restart_modes(v, _keep(), 7, str(tmp_path))
