"""Error-driven adaptive refinement of a sample set.

Triangulate the coarse grid, then insert points where a linear interpolant of
log10(sigma_min) is worst -- the interpolant `plotting.py` contours with, so the
indicator estimates the error actually visible in the plot. Measured against a
dense-SVD reference at equal budget: ~2x better than uniform on west0479,
roughly a wash on nearly-normal operators.

* `triangle_errors` -- per-triangle indicator, `area * spread of values`.
* `refine` -- grow a sample set toward a budget, worst triangles first.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay, QhullError

Sampler = Callable[[NDArray[np.complex128]], NDArray[np.float64]]


def _as_xy(z: NDArray[np.complex128]) -> NDArray[np.float64]:
  return np.column_stack([z.real, z.imag])


def triangle_errors(
  xy: NDArray[np.float64],
  values: NDArray[np.float64],
  simplices: NDArray[np.integer],
) -> NDArray[np.float64]:
  """Per-triangle error indicator: area * spread of `values` over its vertices."""
  v = xy[simplices]
  area = 0.5 * np.abs(
    (v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1])
    - (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1]))
  fv = values[simplices]
  return area * (fv.max(axis=1) - fv.min(axis=1))


def refine(
  points: NDArray[np.complex128],
  sigmin: NDArray[np.float64],
  sample: Sampler,
  budget: int,
  rounds: int,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
  """Grow a sample set toward `budget` points, worst-error triangles first.

  `budget` is a ceiling: one insertion per triangle per round caps each round.
  """
  # One `sample` call per round; degenerate input returns what has been sampled.
  if rounds < 1 or budget <= points.size:
    return points, sigmin

  wanted = budget - points.size
  # The remainder is spread one point at a time over the earliest rounds.
  base, extra = divmod(wanted, rounds)

  for round_index in range(rounds):
    remaining = budget - points.size
    if remaining <= 0 or points.size < 3:
      break

    per_round = base + (1 if round_index < extra else 0)
    if per_round < 1:
      break

    xy = _as_xy(points)
    # log10 keeps the indicator scale-free; sigma_min spans many decades.
    values = np.log10(np.maximum(sigmin, np.finfo(float).tiny))
    try:
      tri = Delaunay(xy)
    except (QhullError, ValueError):
      break
    if tri.simplices.size == 0:
      break

    errors = triangle_errors(xy, values, tri.simplices)
    take = min(per_round, remaining, errors.size)
    worst = np.argsort(errors)[::-1][:take]

    centroids = xy[tri.simplices[worst]].mean(axis=1)
    candidates = np.asarray(
      centroids[:, 0] + 1j * centroids[:, 1], dtype=np.complex128)
    # Keep only centroids not already in the sample set.
    candidates = np.setdiff1d(np.unique(candidates), points)
    if candidates.size == 0:
      break

    points = np.concatenate([points, candidates])
    sigmin = np.concatenate([sigmin, sample(candidates)])

  shortfall = budget - points.size
  if shortfall > 0:
    print(
      f'refinement placed {wanted - shortfall}/{wanted} requested points; '
      f'{shortfall} short because the triangulation ran out of candidates. '
      f'More rounds would place more.',
      flush=True)
  return points, sigmin

