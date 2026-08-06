"""Global assembly and field-block restart output."""

import h5py
import numpy as np
import pytest

from nonmodal.fields import FIELD_BLOCK_COUNT, FIELD_NAMES, write_restart_eigenvectors
from nonmodal.matrices import assemble_global


def _reference_assemble(
  inmat: np.ndarray, nrg: int, ncg: int, lg: np.ndarray
) -> np.ndarray:
  """Plain-numpy reference for `assemble_global`."""
  out = np.zeros((nrg, ncg))
  for i in range(inmat.shape[0]):
    for j in range(inmat.shape[0]):
      out[lg[i], lg[j]] += inmat[i, j]
  return out


def test_assemble_global_matches_reference() -> None:
  rng = np.random.default_rng(0)
  inmat = rng.normal(size=(4, 4))
  lg = np.array([0, 2, 3, 5], dtype=np.int64)

  got = assemble_global(inmat, nrg=6, ncg=6, lg=lg)
  np.testing.assert_allclose(got, _reference_assemble(inmat, 6, 6, lg))


def test_assemble_global_accumulates_duplicate_targets() -> None:
  # Repeated entries in lg must sum rather than overwrite.
  inmat = np.ones((2, 2))
  lg = np.array([1, 1], dtype=np.int64)

  got = assemble_global(inmat, nrg=2, ncg=2, lg=lg)
  assert got[1, 1] == pytest.approx(4.0)
  assert got[0, 0] == pytest.approx(0.0)


def test_assemble_global_rejects_non_square() -> None:
  lg = np.array([0, 1], dtype=np.int64)
  with pytest.raises(ValueError):
    assemble_global(np.ones((2, 3)), nrg=3, ncg=3, lg=lg)


def test_restart_roundtrip_writes_seven_field_blocks(tmp_path) -> None:
  block = 4
  n_global = block * FIELD_BLOCK_COUNT
  keep_global = np.zeros(n_global, dtype=bool)
  keep_global[:block * 5] = True
  nred = int(np.count_nonzero(keep_global))

  eigvecs = (np.arange(nred * 2, dtype=np.complex128).reshape(nred, 2) + 1j)
  write_restart_eigenvectors(eigvecs, keep_global, block * FIELD_BLOCK_COUNT, str(tmp_path))

  for idx in range(2):
    path = tmp_path / f'xmhd2d_{idx:05d}.rst'
    assert path.exists()
    with h5py.File(path, 'r') as f:
      for name in FIELD_NAMES:
        assert name in f, f'missing field block {name}'
        assert f[name].shape == (block,)
      assert f['OFT_idx_Version'][0] == 1
      assert f['t'][0] == float(idx)

    # Dropped blocks stay zero; kept blocks carry the (real part of the) data.
    with h5py.File(path, 'r') as f:
      np.testing.assert_allclose(f[FIELD_NAMES[6]][:], np.zeros(block))
      np.testing.assert_allclose(
        f[FIELD_NAMES[0]][:], eigvecs[:block, idx].real)


def test_restart_rejects_wrong_reduced_size(tmp_path) -> None:
  keep_global = np.ones(FIELD_BLOCK_COUNT * 2, dtype=bool)
  bad = np.zeros((3, 1), dtype=np.complex128)
  with pytest.raises(ValueError, match='do not match reduced size'):
    write_restart_eigenvectors(bad, keep_global, FIELD_BLOCK_COUNT * 2, str(tmp_path))
