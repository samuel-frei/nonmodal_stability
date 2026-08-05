"""Where to sample the complex plane.

Sample sets are flat arrays of complex points -- never meshes. Structure is
reintroduced only at plotting time, by triangulating or interpolating. That
keeps one representation through the whole pipeline instead of two.
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
        'region to cover must be prescribed with strict bounds: '
        '--real-max > --real-min and --imag-max > --imag-min')

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


def near_square(n_points: int) -> tuple[int, int]:
  """Split a point budget into a near-square (nx, ny).

  Rounds rather than factorising: the old factor-pair approach made the shape
  depend on the arithmetic of the total, so 128 became 8x16 and a prime like
  127 became 1x127.
  """
  if n_points < 1:
    raise ValueError('n_points must be >= 1')
  side = max(1, int(round(np.sqrt(n_points))))
  return side, side


def uniform_points(bounds: Bounds, nx: int, ny: int) -> NDArray[np.complex128]:
  """A flat uniform lattice covering `bounds`, including its edges."""
  if nx < 1 or ny < 1:
    raise ValueError('nx and ny must be >= 1')
  x = np.linspace(bounds.real_min, bounds.real_max, nx)
  y = np.linspace(bounds.imag_min, bounds.imag_max, ny)
  X, Y = np.meshgrid(x, y)
  return np.asarray((X + 1j * Y).ravel(), dtype=np.complex128)


def upper_half(bounds: Bounds) -> Bounds:
  """Restrict a box to the closed upper half-plane.

  Used with `mirror_conjugates` when the operator is real. Sampling only
  Im z >= 0 halves the work, and the lower half follows by conjugation.
  """
  return Bounds(bounds.real_min, bounds.real_max, 0.0, max(bounds.imag_max, 1e-12))


def mirror_conjugates(
  z: NDArray[np.complex128], sigmin: NDArray[np.float64]
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
  """Reflect samples across the real axis.

  Valid only when the operator's spectrum is closed under conjugation, i.e.
  when the operator is real: then sigma_min(conj(z)I - A) = sigma_min(zI - A).
  Points already on the real axis are not duplicated.
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


@dataclass(frozen=True)
class RectangularSource:
  """Sample a uniform lattice of `nx` by `ny` points over a box.

  The shape is explicit rather than derived from a total, so a caller who wants
  a non-square lattice over a non-square region can ask for one. `near_square`
  turns a point budget into a shape at the CLI boundary.
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

  Used when no bounds are given. It cannot build points on its own -- the
  spectrum is not known until the operator has been factorised -- so it becomes
  a `RectangularSource` via `resolve` once eigenvalues are in hand.
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
#: What a run may be configured with. `resolve` turns any of these into a
#: ResolvedSource, so the pipeline never branches on which one it got.
PointSource = RectangularSource | FileSource | SpectrumSource
