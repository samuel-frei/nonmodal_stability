"""Kreiss constant: how much transient growth the non-normality permits.

    K(L) = sup_{Re z > 0} Re(z) ||(zI - L)^-1|| = sup_{x>0, y} x / sigma_min(x + iy)

and the Kreiss matrix theorem brackets transient growth with it:
`K <= sup_t ||e^{tL}|| <= e n K`. Every evaluated point is a rigorous lower bound
on K, so the search can be simple without being unsound.

For a real operator `||(zI - L)^-1||` is symmetric about the real axis, which is
therefore stationary in `Im z`, and the supremum is expected there. So the search
is one-dimensional: a log-spaced scan of `x / sigma_min(x)` along the positive
real axis, then a bounded refinement around the best scan point. Two checks
report whether the expectation held, rather than assuming it:

* the real-axis maximiser is a maximum, not a saddle, in the `Im z` direction;
* a coarse scan of the right half-plane finds nothing larger off the axis.

If the off-axis scan does find a larger value, it is polished locally, reported
as the bound (every evaluated point is a valid lower bound) and flagged, since
then only a local maximum off the axis is guaranteed.

* `kreiss_constant` -- the search and both checks, as a `KreissResult`.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from .pseudospectrum import DEFAULT_MODE_TOL, sample_sigmin, sigmin_with_mode

DEFAULT_AXIS_POINTS = 64
DEFAULT_OFFAXIS_X = 24
DEFAULT_OFFAXIS_Y = 16
#: The real-axis scan runs from this fraction of the spectral abscissa's size...
AXIS_LOWER_FACTOR = 1e-3
#: ...to this multiple of the spectral radius, where x / sigma_min is ~1 anyway.
AXIS_UPPER_FACTOR = 10.0
#: Off-axis points must beat the real-axis bound by this much to count.
OFFAXIS_MARGIN = 1e-3
#: Steps off the axis, as fractions of x*, for the saddle check.
CURVATURE_STEPS = (1e-3, 1e-2, 1e-1)


@dataclass(frozen=True)
class KreissResult:
  """A Kreiss lower bound, where it was found, and how far to trust it."""

  #: The largest x / sigma_min found anywhere: a rigorous lower bound on K.
  kreiss: float
  #: Where that value was found.
  z_star_real: float
  z_star_imag: float
  sigma_min_at_z_star: float
  #: The real-axis maximum, from the scan and refinement.
  axis_kreiss: float
  axis_x_star: float
  #: True when the real-axis maximiser is the last scan point: the supremum is
  #: approached as x -> infinity, as for a (nearly) normal operator, where K -> 1.
  axis_at_upper_end: bool
  #: sigma_min(x* + iy) / sigma_min(x*) for y = CURVATURE_STEPS * x*; all >= 1
  #: means x* is a maximum, not a saddle, in the Im z direction.
  curvature_ratios: tuple[float, ...]
  axis_is_local_max_in_y: bool
  #: The best value from the coarse right-half-plane scan, and where.
  offaxis_kreiss: float
  offaxis_z_real: float
  offaxis_z_imag: float
  #: True when the off-axis scan beat the real axis, so the supremum may be off it.
  offaxis_exceeds_axis: bool
  spectral_abscissa: float
  spectral_radius: float
  dimension: int
  #: sup_t ||e^{tL}|| lies in [kreiss, upper_bound], by the Kreiss matrix theorem.
  upper_bound: float
  axis_scan_x: tuple[float, ...]
  axis_scan_value: tuple[float, ...]

  def describe(self) -> dict[str, Any]:
    """A JSON-ready dictionary."""
    return asdict(self)


def _local_max_bracket(values: NDArray[np.float64]) -> tuple[int, int, int]:
  """Indices (lo, best, hi) around the largest scan value."""
  best = int(np.argmax(values))
  return max(best - 1, 0), best, min(best + 1, values.size - 1)


def _refine_offaxis(
  T: NDArray[np.complexfloating],
  start: complex,
  start_value: float,
  tol: float,
) -> tuple[float, complex]:
  """Climb from an off-axis grid point; keep the start if nothing better is found."""
  def neg_log_value(log_xy: NDArray[np.float64]) -> float:
    z = complex(math.exp(log_xy[0]), math.exp(log_xy[1]))
    sigma, _ = sigmin_with_mode(z, T, tol)
    return -(log_xy[0] - math.log(sigma))

  result = optimize.minimize(
    neg_log_value, np.array([math.log(start.real), math.log(start.imag)]),
    method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-6, 'maxfev': 200})
  value = math.exp(-float(result.fun))
  if value <= start_value:
    return start_value, start
  return value, complex(math.exp(float(result.x[0])), math.exp(float(result.x[1])))


def kreiss_constant(
  T: NDArray[np.complexfloating],
  nprocs: int,
  axis_points: int = DEFAULT_AXIS_POINTS,
  offaxis_x: int = DEFAULT_OFFAXIS_X,
  offaxis_y: int = DEFAULT_OFFAXIS_Y,
  tol: float = DEFAULT_MODE_TOL,
) -> KreissResult:
  """Search for the Kreiss constant of the operator whose Schur factor is `T`.

  The operator must be stable: with an eigenvalue at or right of the imaginary
  axis the resolvent norm is unbounded on Re z > 0 and K is infinite.
  """
  eigvals = np.asarray(np.diag(T), dtype=np.complex128)
  abscissa = float(eigvals.real.max())
  if abscissa >= 0.0:
    raise ValueError(
      f'spectral abscissa {abscissa:.6g} >= 0: the operator is not stable, so the '
      f'Kreiss constant is infinite. Report the growth rate instead.')
  radius = float(np.abs(eigvals).max())
  x_lo = AXIS_LOWER_FACTOR * abs(abscissa)
  x_hi = AXIS_UPPER_FACTOR * radius

  # 1. Real axis: log-spaced scan, then refine the best bracket in log x.
  xs = np.geomspace(x_lo, x_hi, axis_points)
  values = xs / sample_sigmin(xs.astype(np.complex128), T, nprocs, 'Kreiss real-axis scan')
  lo, best, hi = _local_max_bracket(values)
  axis_at_upper_end = best == axis_points - 1

  def neg_log_value(log_x: float) -> float:
    x = math.exp(log_x)
    sigma, _ = sigmin_with_mode(complex(x, 0.0), T, tol)
    return -(log_x - math.log(sigma))

  axis_x, axis_value = float(xs[best]), float(values[best])
  if lo < hi:
    refined = optimize.minimize_scalar(
      neg_log_value, bounds=(math.log(xs[lo]), math.log(xs[hi])), method='bounded',
      options={'xatol': 1e-4})
    if math.exp(-float(refined.fun)) > axis_value:
      axis_x, axis_value = math.exp(float(refined.x)), math.exp(-float(refined.fun))
  axis_sigma = axis_x / axis_value

  # 2. Is the real-axis point a maximum in Im z, or a saddle?
  steps = np.asarray(CURVATURE_STEPS) * axis_x
  probes = np.asarray(axis_x + 1j * steps, dtype=np.complex128)
  off = sample_sigmin(probes, T, nprocs, 'Kreiss curvature check')
  ratios = tuple(float(r) for r in off / axis_sigma)
  is_local_max = all(r >= 1.0 - 1e-6 for r in ratios)

  # 3. Coarse right-half-plane scan for anything larger off the axis.
  y_hi = max(float(np.abs(eigvals.imag).max()), radius) * AXIS_UPPER_FACTOR
  grid_x = np.geomspace(x_lo, x_hi, offaxis_x)
  grid_y = np.geomspace(x_lo, y_hi, offaxis_y)
  grid = (grid_x[None, :] + 1j * grid_y[:, None]).ravel()
  grid_values = grid.real / sample_sigmin(grid, T, nprocs, 'Kreiss off-axis scan')
  k = int(np.argmax(grid_values))
  offaxis_value, offaxis_z = float(grid_values[k]), complex(grid[k])
  exceeds = offaxis_value > axis_value * (1.0 + OFFAXIS_MARGIN)
  if exceeds:
    # The grid only located the peak; polish it locally in log coordinates.
    offaxis_value, offaxis_z = _refine_offaxis(T, offaxis_z, offaxis_value, tol)

  if exceeds:
    kreiss, z_star, sigma_star = offaxis_value, offaxis_z, offaxis_z.real / offaxis_value
  else:
    kreiss, z_star, sigma_star = axis_value, complex(axis_x, 0.0), axis_sigma

  n = int(T.shape[0])
  return KreissResult(
    kreiss=kreiss,
    z_star_real=z_star.real,
    z_star_imag=z_star.imag,
    sigma_min_at_z_star=sigma_star,
    axis_kreiss=axis_value,
    axis_x_star=axis_x,
    axis_at_upper_end=axis_at_upper_end,
    curvature_ratios=ratios,
    axis_is_local_max_in_y=is_local_max,
    offaxis_kreiss=offaxis_value,
    offaxis_z_real=offaxis_z.real,
    offaxis_z_imag=offaxis_z.imag,
    offaxis_exceeds_axis=exceeds,
    spectral_abscissa=abscissa,
    spectral_radius=radius,
    dimension=n,
    upper_bound=math.e * n * kreiss,
    axis_scan_x=tuple(float(x) for x in xs),
    axis_scan_value=tuple(float(v) for v in values),
  )
