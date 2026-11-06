"""Reconnected flux: the observable, and the initial conditions that maximise it.

Energy-optimal initial conditions reward any transient, tearing or not. The
observable here is reconnection itself: the `cos(kx)` Fourier amplitude of
the flux perturbation along the current sheet,

    F(u) = (2 / L_x) int_0^{L_x} psi(x, z_s) cos(k x) dx,   k = 2 pi m / L_x,

which for a single harmonic is half the reconnected flux psi_X - psi_O. It is
linear in the state, `F(u) = c^T u`, so the initial condition of unit norm
that maximises it at time `t` needs no iteration. With `w = e^{t L^T} c`,

    max |c^T e^{tL} u0| / ||u0||_N = ||N^-1 w||_N,   u0 ~ N^-1 w,

by Cauchy-Schwarz in the N inner product: one adjoint solve per target time.
In the norm's coordinates (`y = R u`, `B = R L R^-1 = Z T Z*`) this is
`y0 ~ Z e^{t T*} a` with `a = Z* R^-T c`, and on the resolved subspace `T` is
small, so a whole gain curve `G(t)` costs little more than one point.

`c` comes from the geometry: the sheet `z = z_s` must be a line of element
edges (it is for the Harris meshes, which put a cell boundary at z = 0), so
psi there is the 1D Lagrange interpolant of the nodes on it, and the integral
is exact up to quadrature of `cos`. The functional is divided by the flux
scale `psi0 lam_b`, so gains are dimensionless.

* `sheet_flux_weights` -- `c` on the scalar basis, from node coordinates.
* `FluxFunctional` / `sheet_flux_functional` -- `c` on the reduced state.
* `adjoint_vector` -- `a = Z* R^-T c`, the functional in Schur coordinates.
* `flux_gain_curve` -- `G(t)`, the best flux per unit initial norm.
* `FluxSeed` / `flux_optimal_seed` -- the optimal real initial condition at `t`.
* `flux_response` -- forward prediction of `F(t)` and the norm for a seed.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import expm_multiply

from .adapters import DofCoordinates
from .fields import FIELD_BLOCK_COUNT, PSI_BLOCK_ID, aligned_phase
from .norm import Norm

#: Gauss-Legendre points per element edge. cos(kx) over one edge is smooth, so
#: 12 points integrate it to rounding for any harmonic the mesh resolves.
QUADRATURE_POINTS = 12


def _lagrange_basis(nodes: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:
  """Values of the Lagrange polynomials through `nodes` at `points`, (len(points), len(nodes))."""
  values = np.ones((points.size, nodes.size))
  for j, node in enumerate(nodes):
    for i, other in enumerate(nodes):
      if i != j:
        values[:, j] *= (points - other) / (node - other)
  return values


def sheet_flux_weights(
  coords: DofCoordinates,
  order: int,
  harmonic: int = 1,
  sheet_z: float = 0.0,
) -> NDArray[np.float64]:
  """`c` on the scalar basis: `c @ psi = (2/L_x) int psi(x, z_s) cos(k x) dx`.

  Nonzero only on the nodes of the line `z = sheet_z`. Their count must be a
  multiple of `order`, which is what it is when the line is made of element
  edges, each carrying `order + 1` nodes shared at the vertices.
  """
  if order < 1:
    raise ValueError(f'order must be >= 1, got {order}')
  if harmonic < 1:
    raise ValueError(f'harmonic must be >= 1, got {harmonic}')
  tol = 1e-5 * coords.period
  on_line = np.flatnonzero(np.abs(coords.z - sheet_z) <= tol)
  count = on_line.size
  if count == 0:
    raise ValueError(f'no DOFs lie on z = {sheet_z:g}; it must be a line of element edges')
  if count % order != 0:
    raise ValueError(
      f'{count} DOFs lie on z = {sheet_z:g}, not a multiple of order {order}: '
      f'the line is not made of element edges')

  # Sorting from the seam starts at a vertex: folding put the seam node at x_min.
  on_line = on_line[np.argsort(coords.x[on_line])]
  x_line = coords.x[on_line]
  wavenumber = 2.0 * np.pi * harmonic / coords.period
  gauss, gauss_weights = np.polynomial.legendre.leggauss(QUADRATURE_POINTS)

  weights = np.zeros(coords.x.size)
  for start in range(0, count, order):
    local = np.arange(start, start + order + 1) % count
    nodes = x_line[local].copy()
    if start + order >= count:
      nodes[-1] += coords.period  # the last edge closes onto the seam vertex
    if np.any(np.diff(nodes) <= 0.0):
      raise ValueError(f'nodes of the edge at x = {nodes[0]:g} are not increasing')
    half = 0.5 * (nodes[-1] - nodes[0])
    points = nodes[0] + half * (gauss + 1.0)
    quadrature = gauss_weights * half * np.cos(wavenumber * points)
    integrand = quadrature @ _lagrange_basis(nodes, points)
    np.add.at(weights, on_line[local], integrand)
  return weights * (2.0 / coords.period)


@dataclass(frozen=True)
class FluxFunctional:
  """The sheet-flux observable as a vector on the reduced state, `F(u) = c @ u`."""

  #: Reduced-basis weights, already divided by `scale`.
  weights: NDArray[np.float64] = field(repr=False)
  harmonic: int
  sheet_z: float
  period: float
  #: What F is divided by [T m], so that it is dimensionless.
  scale: float
  #: DOFs on the sheet line.
  n_nodes: int

  @property
  def wavenumber(self) -> float:
    """k = 2 pi m / L_x [1/m]."""
    return 2.0 * np.pi * self.harmonic / self.period

  def describe(self) -> dict[str, Any]:
    """A JSON-ready record of what was measured."""
    return {
      'observable': '(2/L_x) int psi(x, z_s) cos(k x) dx / scale; for one harmonic, '
                    'half the reconnected flux psi_X - psi_O',
      'harmonic': int(self.harmonic),
      'wavenumber_per_m': float(self.wavenumber),
      'sheet_z_m': float(self.sheet_z),
      'period_m': float(self.period),
      'scale_T_m': float(self.scale),
      'nodes_on_sheet': int(self.n_nodes),
    }


def sheet_flux_functional(
  coords: DofCoordinates,
  order: int,
  keep_global: NDArray[np.bool_],
  scale: float,
  harmonic: int = 1,
  sheet_z: float = 0.0,
) -> FluxFunctional:
  """The sheet flux on the reduced state: `c` in the psi block, zero elsewhere."""
  n_scalar = coords.x.size
  if keep_global.size != FIELD_BLOCK_COUNT * n_scalar:
    raise ValueError(
      f'reduction mask has {keep_global.size} entries, expected '
      f'{FIELD_BLOCK_COUNT} x {n_scalar}')
  if scale <= 0.0:
    raise ValueError(f'scale must be positive, got {scale}')
  scalar = sheet_flux_weights(coords, order, harmonic, sheet_z)

  full = np.zeros(keep_global.size)
  full[PSI_BLOCK_ID * n_scalar:(PSI_BLOCK_ID + 1) * n_scalar] = scalar
  dropped = (full != 0.0) & ~keep_global
  if dropped.any():
    raise ValueError(f'{int(dropped.sum())} sheet DOFs are not in the reduced state')
  return FluxFunctional(
    weights=full[keep_global] / scale,
    harmonic=harmonic,
    sheet_z=sheet_z,
    period=coords.period,
    scale=scale,
    n_nodes=int(np.count_nonzero(scalar)),
  )


def adjoint_vector(
  functional: FluxFunctional,
  schur_z: NDArray[np.complexfloating],
  norm: Norm,
) -> NDArray[np.complex128]:
  """`a = Z* R^-T c`, so that `F(u) = a* q` for `u = R^-1 Z q`.

  `schur_z` may be tall (`Z1` of a restriction); then `a` lives on the subspace.
  """
  dual = norm.dual(functional.weights)
  if schur_z.shape[0] != dual.size:
    raise ValueError(f'Schur vectors have {schur_z.shape[0]} rows, the functional {dual.size}')
  return np.asarray(schur_z.conj().T @ dual, dtype=np.complex128)


def _propagate(
  matrix: NDArray[np.complexfloating],
  start: NDArray[np.complexfloating],
  times: Sequence[float] | NDArray[np.float64],
) -> NDArray[np.complex128]:
  """`e^{t M} start` for each `t`, (len(times), n), stepping between sorted times."""
  t = np.asarray(times, dtype=np.float64)
  if t.ndim != 1 or t.size == 0:
    raise ValueError('times must be a non-empty 1-D sequence')
  if (t < 0.0).any():
    raise ValueError('times must be >= 0')
  order = np.argsort(t)
  out = np.empty((t.size, start.size), dtype=np.complex128)
  current = np.asarray(start, dtype=np.complex128)
  now = 0.0
  for i in order:
    if t[i] > now:
      current = expm_multiply((t[i] - now) * matrix, current)
      now = float(t[i])
    out[i] = current
  return out


def flux_gain_curve(
  schur_t: NDArray[np.complexfloating],
  adjoint: NDArray[np.complexfloating],
  times: Sequence[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
  """`G(t) = ||e^{t T*} a||`: the largest `|F(t)|` from a unit initial condition."""
  return np.asarray(
    np.linalg.norm(_propagate(schur_t.conj().T, adjoint, times), axis=1), dtype=np.float64)


@dataclass(frozen=True)
class FluxSeed:
  """The real initial condition of unit norm that maximises F at one time."""

  target_time: float
  #: G(t): the optimum over all (complex) initial conditions in the subspace.
  gain: float
  #: F(t) from the real seed; equal to `gain` when the subspace is conjugate-closed.
  achieved: float
  #: ||Im|| / ||y0|| of the optimum before it was made real; ~0 for a real operator.
  imag_fraction: float
  #: The seed in Schur coordinates, for forward predictions.
  schur_vector: NDArray[np.complex128] = field(repr=False)
  #: The seed as reduced finite-element coefficients, unit in the norm.
  vector: NDArray[np.float64] = field(repr=False)

  def describe(self) -> dict[str, Any]:
    """JSON-safe summary, without the vectors."""
    return {
      'target_time_s': float(self.target_time),
      'gain': float(self.gain),
      'achieved': float(self.achieved),
      'imag_fraction': float(self.imag_fraction),
    }


def flux_optimal_seed(
  schur_t: NDArray[np.complexfloating],
  schur_z: NDArray[np.complexfloating],
  adjoint: NDArray[np.complexfloating],
  target_time: float,
  norm: Norm,
) -> FluxSeed:
  """The unit-norm real initial condition that maximises F at `target_time`.

  The complex optimum is `y0 = Z e^{t T*} a / G`. For a real operator and a
  conjugate-closed subspace it is real up to one global phase, which is
  removed; any remainder is recorded rather than hidden. The sign is chosen so
  that F(t) > 0.
  """
  w = _propagate(schur_t.conj().T, adjoint, [target_time])[0]
  gain = float(np.linalg.norm(w))
  if gain == 0.0:
    raise ValueError('the observable is blind to this subspace: G(t) = 0')
  y0 = schur_z @ (w / gain)
  rotated = y0 * np.exp(-1j * aligned_phase(y0))
  imag_fraction = float(np.linalg.norm(rotated.imag) / np.linalg.norm(rotated))
  real = np.asarray(rotated.real, dtype=np.float64)
  real /= np.linalg.norm(real)

  # The real vector lies in the span (exactly, for a conjugate-closed subspace),
  # so its Schur coordinates are a projection.
  q = np.asarray(schur_z.conj().T @ real, dtype=np.complex128)
  achieved = float(np.real(np.vdot(adjoint, _propagate(schur_t, q, [target_time])[0])))
  if achieved < 0.0:
    real, q, achieved = -real, -q, -achieved
  return FluxSeed(
    target_time=float(target_time),
    gain=gain,
    achieved=achieved,
    imag_fraction=imag_fraction,
    schur_vector=q,
    vector=np.asarray(norm.to_basis(real), dtype=np.float64),
  )


def flux_response(
  schur_t: NDArray[np.complexfloating],
  adjoint: NDArray[np.complexfloating],
  schur_vector: NDArray[np.complexfloating],
  times: Sequence[float] | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Forward prediction for a seed: `F(t)` and `||u(t)||_N`, exact in time."""
  states = _propagate(schur_t, schur_vector, times)
  flux = np.real(states @ adjoint.conj())
  return np.asarray(flux, dtype=np.float64), np.asarray(
    np.linalg.norm(states, axis=1), dtype=np.float64)
