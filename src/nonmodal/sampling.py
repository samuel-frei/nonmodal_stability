"""Where to sample the complex plane.

Sample sets are flat arrays of complex points, never meshes; structure returns
only at plotting time. The initial grid is coarse and always derived from the
spectrum, since this tool refines onto features rather than sweeping a chosen
box, so resolution is asked for as `nx` by `ny` and not as a point total.

* `Bounds` -- an axis-aligned box, with constructors from a spectrum or points.
* `uniform_points` -- a flat lattice covering a box, including its edges.
* `mirror_conjugates` -- reflect samples across the real axis.
* `load_flat_grid_npy` -- read a flat complex point set from `.npy`.
* `RectangularSource` / `FileSource` / `SpectrumSource` -- point sources; each
  has `resolve(eigvals)`, so the pipeline never branches on which it got.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Bounds:
  """An axis-aligned box in the complex plane."""

  real_min: float
  real_max: float
  imag_min: float
  imag_max: float

  def __post_init__(self) -> None:
    if self.real_max <= self.real_min or self.imag_max <= self.imag_min:
      raise ValueError(
        f'bounds must be strictly ordered, got real [{self.real_min}, '
        f'{self.real_max}], imag [{self.imag_min}, {self.imag_max}]')

  @classmethod
  def around_spectrum(
    cls, eigvals: NDArray[np.complexfloating], pad: float = 0.3
  ) -> Bounds:
    """A padded box enclosing a spectrum."""
    finite = eigvals[np.isfinite(eigvals.real) & np.isfinite(eigvals.imag)]
    if finite.size == 0:
      raise ValueError('no finite eigenvalues to bound')
    span = max(
      float(finite.real.max() - finite.real.min()),
      2.0 * float(np.abs(finite.imag).max()),
      1e-12)
    p = pad * span
    imag_extent = float(np.abs(finite.imag).max()) + p
    return cls(float(finite.real.min()) - p, float(finite.real.max()) + p,
               -imag_extent, imag_extent)

  @classmethod
  def around_points(cls, z: NDArray[np.complex128]) -> Bounds:
    """The tightest box containing a set of sample points."""
    return cls(float(z.real.min()), float(z.real.max()),
               float(z.imag.min()), float(z.imag.max()))

  def as_dict(self) -> dict[str, float]:
    return {
      'real_min': self.real_min, 'real_max': self.real_max,
      'imag_min': self.imag_min, 'imag_max': self.imag_max,
    }


def _axis_through_zero(lo: float, hi: float, n: int) -> NDArray[np.float64]:
  """`n` points spanning [lo, hi], with an exact 0.0 when the span straddles it."""
  # A plain linspace steps over zero for even n and can land on ~9e-16 for odd n;
  # both break the exact comparisons downstream. See the CLAUDE.md landmine.
  if lo >= 0.0 or hi <= 0.0:
    return np.linspace(lo, hi, n)
  if n == 1:
    return np.zeros(1)

  # Split the interior points between the two sides in proportion to their
  # widths, so the spacing stays as even as the counts allow.
  interior = n - 1
  below = int(round(interior * (-lo) / (hi - lo)))
  below = min(max(below, 1), interior - 1) if interior >= 2 else interior
  above = interior - below

  negative = np.linspace(lo, 0.0, below + 1)[:-1] if below else np.empty(0)
  positive = np.linspace(0.0, hi, above + 1)[1:] if above else np.empty(0)
  return np.concatenate([negative, np.zeros(1), positive])


def uniform_points(bounds: Bounds, nx: int, ny: int) -> NDArray[np.complex128]:
  """A flat uniform lattice covering `bounds`, including its edges.

  `Im z = 0` is always a row when the region straddles it; contours pinch there.
  """
  if nx < 1 or ny < 1:
    raise ValueError('nx and ny must be >= 1')
  x = np.linspace(bounds.real_min, bounds.real_max, nx)
  y = _axis_through_zero(bounds.imag_min, bounds.imag_max, ny)
  X, Y = np.meshgrid(x, y)
  return np.asarray((X + 1j * Y).ravel(), dtype=np.complex128)


def mirror_conjugates(
  z: NDArray[np.complex128], sigmin: NDArray[np.float64]
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
  """Reflect samples across the real axis, without duplicating the axis itself.

  Valid only for a real operator, whose spectrum is closed under conjugation.
  """
  off_axis = z.imag != 0.0
  return (
    np.concatenate([z, np.conj(z[off_axis])]),
    np.concatenate([sigmin, sigmin[off_axis]]))


def load_flat_grid_npy(path: str) -> NDArray[np.complex128]:
  """Load a flat complex point set from a .npy file."""
  zz = np.load(path, allow_pickle=False)
  if zz.size == 0:
    raise ValueError('grid-npy must contain at least one complex value')
  zz = np.asarray(zz).ravel()
  if not np.iscomplexobj(zz):
    zz = zz.astype(np.complex128)
  return np.asarray(zz, dtype=np.complex128)


DEFAULT_BOUNDS_PAD = 0.3
DEFAULT_GRID_NX = 25
#: Odd, so the zero row is a lattice point and the spacing stays even.
DEFAULT_GRID_NY = 25


@dataclass(frozen=True)
class RectangularSource:
  """Sample a uniform lattice of `nx` by `ny` points over a box.

  Dimensions are explicit; a total would have to be factored back into a shape.
  """

  bounds: Bounds
  nx: int
  ny: int

  def __post_init__(self) -> None:
    if self.nx < 1 or self.ny < 1:
      raise ValueError('grid-nx and grid-ny must be >= 1')

  @property
  def n_points(self) -> int:
    return self.nx * self.ny

  def resolve(self, eigvals: NDArray[np.complexfloating]) -> RectangularSource:
    return self

  def build(self) -> NDArray[np.complex128]:
    return uniform_points(self.bounds, self.nx, self.ny)

  def describe(self) -> dict[str, object]:
    return {'kind': 'rectangular', 'n_points': int(self.n_points),
            'nx': int(self.nx), 'ny': int(self.ny),
            'bounds': self.bounds.as_dict()}


@dataclass(frozen=True)
class FileSource:
  """Sample an externally supplied point set."""

  path: str

  def resolve(self, eigvals: NDArray[np.complexfloating]) -> FileSource:
    return self

  def build(self) -> NDArray[np.complex128]:
    return load_flat_grid_npy(self.path)

  def describe(self) -> dict[str, object]:
    return {'kind': 'file', 'path': self.path}


@dataclass(frozen=True)
class SpectrumSource:
  """Sample a box inferred from the operator's own spectrum.

  Cannot build points alone; `resolve` turns it into a `RectangularSource`.
  """

  nx: int
  ny: int
  pad: float = DEFAULT_BOUNDS_PAD

  def __post_init__(self) -> None:
    if self.nx < 1 or self.ny < 1:
      raise ValueError('grid-nx and grid-ny must be >= 1')
    if self.pad < 0.0:
      raise ValueError('bounds-pad must be >= 0')

  @property
  def n_points(self) -> int:
    return self.nx * self.ny

  def resolve(self, eigvals: NDArray[np.complexfloating]) -> RectangularSource:
    bounds = Bounds.around_spectrum(eigvals, pad=self.pad)
    print(
      f'inferred sampling region from the spectrum (pad={self.pad:g}): '
      f'Re[z] in [{bounds.real_min:.6g}, {bounds.real_max:.6g}], '
      f'Im[z] in [{bounds.imag_min:.6g}, {bounds.imag_max:.6g}]',
      flush=True)
    return RectangularSource(bounds, self.nx, self.ny)

  def describe(self) -> dict[str, object]:
    return {'kind': 'spectrum', 'n_points': int(self.n_points),
            'nx': int(self.nx), 'ny': int(self.ny), 'pad': float(self.pad)}


#: A source that can already produce points.
ResolvedSource = RectangularSource | FileSource
#: What a run may be configured with; `resolve` maps any of these to the above.
PointSource = RectangularSource | FileSource | SpectrumSource
