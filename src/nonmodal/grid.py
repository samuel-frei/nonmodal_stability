"""Sampling-grid geometry helpers for the complex plane."""


import numpy as np
from numpy.typing import NDArray


def _grid_shape(total_points: int) -> tuple[int, int]:
  """Choose the factor pair (rows, cols) that is as square as possible."""
  n = int(total_points)
  rows = int(np.floor(np.sqrt(n)))
  while rows > 1 and (n % rows != 0):
    rows -= 1
  cols = n // rows
  return rows, cols


def _is_real_axis_symmetric_grid(c: NDArray[np.float64]) -> bool:
  """Return True when sampled imaginary coordinates are symmetric about zero."""
  c = np.asarray(c, dtype=float)
  scale = max(1.0, float(np.max(np.abs(c))))
  atol = max(1e-12, np.finfo(float).eps * scale * 32.0)
  return bool(np.allclose(c, -c[::-1], rtol=0.0, atol=atol))


def load_flat_grid_npy(path: str) -> NDArray[np.complex128]:
  """Load a flat complex grid from a .npy file."""
  zz = np.load(path, allow_pickle=False)
  if zz.size == 0:
    raise ValueError('grid-npy must contain at least one complex value')
  zz = np.asarray(zz).ravel()
  if not np.iscomplexobj(zz):
    zz = zz.astype(np.complex128)
  return zz


def grid_bounds_from_flat(
  zz_flat: NDArray[np.complex128],
) -> tuple[float, float, float, float]:
  """Compute real/imaginary bounds from a flat complex grid."""
  real_min = float(np.min(zz_flat.real))
  real_max = float(np.max(zz_flat.real))
  imag_min = float(np.min(zz_flat.imag))
  imag_max = float(np.max(zz_flat.imag))
  return real_min, real_max, imag_min, imag_max
