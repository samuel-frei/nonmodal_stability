"""Point-set construction, bounds, and conjugate mirroring."""

import numpy as np
import pytest

from nonmodal.sampling import (
  Bounds,
  FileSource,
  RectangularSource,
  load_flat_grid_npy,
  mirror_conjugates,
  near_square,
  uniform_points,
  upper_half,
)


@pytest.mark.parametrize(
  ('total', 'expected'),
  [(16, (4, 4)), (100, (10, 10)), (128, (11, 11)), (127, (11, 11)), (1, (1, 1))],
)
def test_near_square_rounds_instead_of_factoring(
  total: int, expected: tuple[int, int]
) -> None:
  # The old _grid_shape factorised, so 128 -> 8x16 and a prime 127 -> 1x127.
  assert near_square(total) == expected


def test_near_square_rejects_empty() -> None:
  with pytest.raises(ValueError, match='n_points must be >= 1'):
    near_square(0)


def test_bounds_reject_inverted() -> None:
  with pytest.raises(ValueError, match='strict bounds'):
    Bounds(1.0, -1.0, -1.0, 1.0)
  with pytest.raises(ValueError, match='strict bounds'):
    Bounds(-1.0, 1.0, 1.0, -1.0)


def test_bounds_around_spectrum_encloses_it() -> None:
  eigvals = np.array([1 + 2j, -3 - 1j, 0 + 0j], dtype=np.complex128)
  b = Bounds.around_spectrum(eigvals)
  assert b.real_min < eigvals.real.min()
  assert b.real_max > eigvals.real.max()
  assert b.imag_min < eigvals.imag.min()
  assert b.imag_max > eigvals.imag.max()
  # Imaginary bounds stay symmetric so half-plane sampling stays balanced.
  assert b.imag_min == pytest.approx(-b.imag_max)


def test_bounds_around_spectrum_ignores_nonfinite() -> None:
  eigvals = np.array([1 + 1j, np.nan + 0j, np.inf + 0j], dtype=np.complex128)
  b = Bounds.around_spectrum(eigvals)
  assert np.isfinite([b.real_min, b.real_max, b.imag_min, b.imag_max]).all()


def test_bounds_around_points_is_tight() -> None:
  z = np.array([-1 - 1j, 2 + 3j], dtype=np.complex128)
  b = Bounds.around_points(z)
  assert (b.real_min, b.real_max, b.imag_min, b.imag_max) == (-1.0, 2.0, -1.0, 3.0)


def test_uniform_points_covers_corners() -> None:
  b = Bounds(-1.0, 1.0, -2.0, 2.0)
  z = uniform_points(b, 3, 5)
  assert z.size == 15
  assert z.real.min() == pytest.approx(-1.0)
  assert z.real.max() == pytest.approx(1.0)
  assert z.imag.min() == pytest.approx(-2.0)
  assert z.imag.max() == pytest.approx(2.0)


def test_upper_half_clips_to_nonnegative_imaginary() -> None:
  b = upper_half(Bounds(-1.0, 1.0, -2.0, 2.0))
  assert b.imag_min == 0.0
  assert b.imag_max == 2.0


def test_mirror_conjugates_reflects_without_duplicating_the_axis() -> None:
  z = np.array([1 + 1j, 2 + 0j], dtype=np.complex128)
  s = np.array([10.0, 20.0])
  mz, ms = mirror_conjugates(z, s)

  # Only the one off-axis point is duplicated; the real-axis point is not.
  assert mz.size == 3
  assert ms.size == 3
  assert complex(1, -1) in set(mz.tolist())
  assert ms[-1] == pytest.approx(10.0)


def test_rectangular_source_builds_its_budget() -> None:
  src = RectangularSource(Bounds(-1.0, 1.0, -1.0, 1.0), 25)
  z = src.build()
  assert z.size == 25
  assert src.describe()['kind'] == 'rectangular'


def test_file_source_round_trip(tmp_path) -> None:
  z = np.array([1 + 2j, -3 + 4j], dtype=np.complex128)
  path = tmp_path / 'grid.npy'
  np.save(path, z)

  src = FileSource(str(path))
  np.testing.assert_allclose(src.build(), z)
  assert src.describe()['kind'] == 'file'


def test_flat_grid_promotes_real_input(tmp_path) -> None:
  path = tmp_path / 'real.npy'
  np.save(path, np.array([1.0, 2.0, 3.0]))
  assert np.iscomplexobj(load_flat_grid_npy(str(path)))


def test_flat_grid_rejects_empty(tmp_path) -> None:
  path = tmp_path / 'empty.npy'
  np.save(path, np.array([], dtype=np.complex128))
  with pytest.raises(ValueError, match='at least one complex value'):
    load_flat_grid_npy(str(path))
