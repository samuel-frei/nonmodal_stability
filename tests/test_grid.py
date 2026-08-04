"""Grid geometry and flat-grid IO."""

import numpy as np
import pytest

from nonmodal.grid import (
  _grid_shape,
  _is_real_axis_symmetric_grid,
  grid_bounds_from_flat,
  load_flat_grid_npy,
)


@pytest.mark.parametrize(
  ('total', 'expected'),
  [
    (16, (4, 4)),
    (128, (8, 16)),
    (100, (10, 10)),
    (12, (3, 4)),
    (1, (1, 1)),
    (7, (1, 7)),  # prime: no square-ish factor pair exists
  ],
)
def test_grid_shape_factors_exactly(total: int, expected: tuple[int, int]) -> None:
  rows, cols = _grid_shape(total)
  assert (rows, cols) == expected
  assert rows * cols == total


def test_symmetric_grid_detected() -> None:
  assert _is_real_axis_symmetric_grid(np.linspace(-3.0, 3.0, 7))
  assert _is_real_axis_symmetric_grid(np.linspace(-3.0, 3.0, 4))


def test_asymmetric_grid_rejected() -> None:
  assert not _is_real_axis_symmetric_grid(np.linspace(-3.0, 2.5, 7))
  assert not _is_real_axis_symmetric_grid(np.linspace(0.0, 3.0, 7))


def test_flat_grid_roundtrip(tmp_path) -> None:
  zz = np.array([1 + 2j, -3 + 4j, 0.5 - 1j], dtype=np.complex128)
  path = tmp_path / 'grid.npy'
  np.save(path, zz)

  loaded = load_flat_grid_npy(str(path))
  np.testing.assert_allclose(loaded, zz)

  assert grid_bounds_from_flat(loaded) == (-3.0, 1.0, -1.0, 4.0)


def test_flat_grid_promotes_real_input(tmp_path) -> None:
  path = tmp_path / 'real.npy'
  np.save(path, np.array([1.0, 2.0, 3.0]))

  loaded = load_flat_grid_npy(str(path))
  assert np.iscomplexobj(loaded)


def test_flat_grid_rejects_empty(tmp_path) -> None:
  path = tmp_path / 'empty.npy'
  np.save(path, np.array([], dtype=np.complex128))

  with pytest.raises(ValueError, match='at least one complex value'):
    load_flat_grid_npy(str(path))
