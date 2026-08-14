"""Run metadata and flat sample IO, checked by round-tripping what a run writes."""

import json

import numpy as np
import pytest

from nonmodal.config import RunConfig
from nonmodal.io import (
  EIGVALS_FILE,
  build_metadata,
  load_samples,
  read_eigenmodes,
  read_metadata,
  read_pseudomodes,
  save_samples,
  write_eigenmodes,
  write_metadata,
  write_pseudomodes,
)
from nonmodal.sampling import Bounds, SpectrumSource


def _samples() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  z = np.array([0 + 0j, 1 + 2j, -3 - 0.5j], dtype=np.complex128)
  return z, np.array([1e-3, 4.5e-7, 2.0]), np.array([1 + 1j, -2 + 0j], dtype=np.complex128)


def test_samples_round_trip(tmp_path) -> None:
  z, sigmin, eigvals = _samples()
  save_samples(str(tmp_path), z, sigmin, eigvals)
  got_z, got_sigmin, got_eigvals = load_samples(str(tmp_path))

  np.testing.assert_array_equal(got_z, z)
  np.testing.assert_array_equal(got_sigmin, sigmin)
  np.testing.assert_array_equal(got_eigvals, eigvals)


def test_samples_load_without_a_spectrum_file(tmp_path) -> None:
  """A run predating the eigenvalue overlay still loads, with an empty spectrum."""
  z, sigmin, eigvals = _samples()
  save_samples(str(tmp_path), z, sigmin, eigvals)
  (tmp_path / EIGVALS_FILE).unlink()

  got_z, _, got_eigvals = load_samples(str(tmp_path))
  assert got_eigvals.size == 0
  np.testing.assert_array_equal(got_z, z)


def test_mismatched_points_and_values_are_rejected(tmp_path) -> None:
  z, sigmin, eigvals = _samples()
  save_samples(str(tmp_path), z, sigmin, eigvals)
  np.save(tmp_path / 'pseudo_sigmin.npy', np.ones(z.size + 1))

  with pytest.raises(ValueError, match='disagree'):
    load_samples(str(tmp_path))


def test_metadata_round_trip(tmp_path) -> None:
  config = RunConfig(source=SpectrumSource(nx=4, ny=4), run_tag='r', case_tag='c')
  metadata = build_metadata(
    config, Bounds(-1.0, 1.0, -2.0, 2.0),
    n_points=16, n_evaluated=9, half_plane=True, effective_workers=4)

  write_metadata(str(tmp_path), metadata)
  got = read_metadata(str(tmp_path))

  assert got == metadata
  assert got['sampling']['points_evaluated'] == 9
  assert got['sampling']['half_plane_symmetry'] is True
  assert got['sampling']['bounds']['imag_max'] == pytest.approx(2.0)


def test_metadata_is_json_serialisable_as_written(tmp_path) -> None:
  """`build_metadata` holds numpy-free scalars, so the sidecar needs no encoder."""
  config = RunConfig(source=SpectrumSource(nx=4, ny=4))
  metadata = build_metadata(
    config, Bounds.around_points(_samples()[0]),
    n_points=3, n_evaluated=3, half_plane=False, effective_workers=1)

  write_metadata(str(tmp_path), metadata)
  assert json.loads((tmp_path / 'run_metadata.json').read_text()) == metadata


def test_pseudomode_sidecar_round_trip(tmp_path) -> None:
  payload = {'run_tag': 'r', 'phases': 8, 'tol': 1e-10,
             'modes': [{'z_real': 5e5, 'z_imag': -2.4e4, 'sigma_min': 3.2e-7}]}
  write_pseudomodes(str(tmp_path), payload)
  assert read_pseudomodes(str(tmp_path)) == payload


def test_eigenmode_sidecar_round_trip(tmp_path) -> None:
  """The sidecar is what names which eigenvalue each restart file holds."""
  payload = {'run_tag': 'r', 'case_tag': 'c', 'modes': [
    {'eigenvalue_real': -27.9045, 'eigenvalue_imag': 5.8e-9, 'file': 'xmhd2d_00000.rst'},
    {'eigenvalue_real': -27.8600, 'eigenvalue_imag': -5.8e-9, 'file': 'xmhd2d_00001.rst'}]}
  write_eigenmodes(str(tmp_path), payload)
  got = read_eigenmodes(str(tmp_path))

  assert got == payload
  # The conjugate twins stay distinguishable by eigenvalue, not by filename.
  assert len({m['eigenvalue_real'] for m in got['modes']}) == 2


def test_sidecars_create_a_missing_directory(tmp_path) -> None:
  nested = tmp_path / 'run' / 'eigvecs_plot'
  write_eigenmodes(str(nested), {'modes': []})
  assert read_eigenmodes(str(nested)) == {'modes': []}
