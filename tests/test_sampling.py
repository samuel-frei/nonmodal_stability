"""Point-set construction, bounds, and conjugate mirroring."""

import numpy as np
import pytest

from nonmodal.sampling import (
  Bounds,
  FileSource,
  RectangularSource,
  SpectrumSource,
  load_flat_grid_npy,
  mirror_conjugates,
  uniform_points,
)


def test_bounds_reject_inverted() -> None:
  with pytest.raises(ValueError, match='strictly ordered'):
    Bounds(1.0, -1.0, -1.0, 1.0)
  with pytest.raises(ValueError, match='strictly ordered'):
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


@pytest.mark.parametrize('ny', [1, 2, 3, 4, 5, 8, 23, 24, 25, 40])
def test_real_axis_is_always_sampled(ny: int) -> None:
  """Im z = 0 must be an exact sample row whenever the region straddles it.

  A real operator's pseudospectrum is symmetric about the real axis and its
  contours pinch there. A plain linspace steps over zero for even ny, and for
  some odd ny lands on ~9e-16 instead of exactly 0.0.
  """
  z = uniform_points(Bounds(-1.0, 1.0, -6.4, 6.4), nx=3, ny=ny)
  rows = np.unique(z.imag)

  assert rows.size == ny, 'the requested row count must be honoured exactly'
  assert np.count_nonzero(rows == 0.0) == 1, 'exactly one row must be exact zero'


@pytest.mark.parametrize('ny', [3, 5, 25])
def test_odd_row_counts_keep_even_spacing(ny: int) -> None:
  """With an odd count the axis is a lattice point, so nothing is distorted."""
  rows = np.unique(uniform_points(Bounds(-1.0, 1.0, -6.4, 6.4), nx=2, ny=ny).imag)
  spacing = np.diff(rows)
  np.testing.assert_allclose(spacing, spacing[0])


def test_region_not_straddling_the_axis_is_untouched() -> None:
  """The zero-splicing must not perturb a box that never crosses the axis."""
  z = uniform_points(Bounds(-1.0, 1.0, 0.5, 2.0), nx=3, ny=5)
  np.testing.assert_allclose(np.unique(z.imag), np.linspace(0.5, 2.0, 5))


@pytest.mark.parametrize('ny', [4, 23, 24, 25])
def test_half_plane_then_mirror_produces_no_near_duplicates(ny: int) -> None:
  """A near-zero row would be mirrored into a pair ~1e-15 apart.

  That pair is both a wasted evaluation and a degenerate triangulation waiting
  to happen, so the axis row has to be exactly zero.
  """
  z = uniform_points(Bounds(-1.0, 1.0, -6.4, 6.4), nx=3, ny=ny)
  upper = z[z.imag >= 0.0]
  mirrored, _ = mirror_conjugates(upper, np.ones(upper.size))

  assert np.unique(mirrored).size == mirrored.size, 'mirroring duplicated a point'
  rows = np.unique(mirrored.imag)
  assert np.min(np.diff(rows)) > 1e-6, 'two rows collapsed onto the axis'
  # The mirrored set is symmetric about the axis.
  np.testing.assert_allclose(np.sort(rows), -np.sort(-rows)[::-1])


def test_mirror_conjugates_reflects_without_duplicating_the_axis() -> None:
  z = np.array([1 + 1j, 2 + 0j], dtype=np.complex128)
  s = np.array([10.0, 20.0])
  mz, ms = mirror_conjugates(z, s)

  # Only the one off-axis point is duplicated; the real-axis point is not.
  assert mz.size == 3
  assert ms.size == 3
  assert complex(1, -1) in set(mz.tolist())
  assert ms[-1] == pytest.approx(10.0)


def test_rectangular_source_builds_its_lattice() -> None:
  src = RectangularSource(Bounds(-1.0, 1.0, -1.0, 1.0), 5, 5)
  z = src.build()
  assert z.size == src.n_points == 25
  assert src.describe()['kind'] == 'rectangular'


def test_rectangular_source_allows_non_square_lattices() -> None:
  src = RectangularSource(Bounds(-4.0, 4.0, -1.0, 1.0), nx=16, ny=4)
  assert src.build().size == src.n_points == 64


def test_rectangular_source_rejects_empty_lattice() -> None:
  with pytest.raises(ValueError, match='grid-nx and grid-ny must be >= 1'):
    RectangularSource(Bounds(-1.0, 1.0, -1.0, 1.0), 0, 5)


def test_spectrum_source_resolves_against_a_spectrum() -> None:
  eigvals = np.array([1 + 2j, -3 - 1j], dtype=np.complex128)
  src = SpectrumSource(nx=6, ny=6, pad=0.5)
  assert src.describe()['kind'] == 'spectrum'

  resolved = src.resolve(eigvals)
  assert isinstance(resolved, RectangularSource)
  assert (resolved.nx, resolved.ny) == (6, 6)
  # The inferred box encloses the spectrum it was derived from.
  assert resolved.bounds.real_min < eigvals.real.min()
  assert resolved.bounds.real_max > eigvals.real.max()


def test_concrete_sources_resolve_to_themselves() -> None:
  eigvals = np.array([1 + 0j], dtype=np.complex128)
  rect = RectangularSource(Bounds(-1.0, 1.0, -1.0, 1.0), 3, 3)
  assert rect.resolve(eigvals) is rect
  assert FileSource('g.npy').resolve(eigvals) == FileSource('g.npy')


def test_spectrum_source_rejects_negative_pad() -> None:
  with pytest.raises(ValueError, match='bounds-pad must be >= 0'):
    SpectrumSource(nx=4, ny=4, pad=-0.1)


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
