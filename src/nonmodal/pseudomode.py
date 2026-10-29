"""Pseudomodes: the vector a near-singular resolvent actually amplifies.

The pseudomode at `z` is the right singular vector of `zI - A` belonging to
`sigma_min`, so it satisfies `||(A - zI)v|| = sigma_min`: an approximate
eigenvector with a known residual. It comes out of the same inverse iteration
the sampler runs -- which discards it -- and reaches the physical basis as
`v_A = Z v_T`, since sigma_min is invariant under `A = Z T Z*` but its singular
vectors are not. In a weighted norm `A` is `B = R L R^-1`, and the mode takes one
more step, `x = R^-1 v_A`, to reach finite-element coefficients.

* `Pseudomode` -- one mode: the point, sigma_min, the vector, the residual.
* `pseudomode_at` -- extract the mode at a point, in the physical basis.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .norm import L2_COEFFICIENTS, Norm
from .pseudospectrum import DEFAULT_MODE_TOL, sigmin_with_mode


@dataclass(frozen=True)
class Pseudomode:
  """One pseudomode, with the evidence that it is one."""

  z: complex
  sigma_min: float
  #: Coefficients in the reduced basis that `keep_global` indexes; a unit
  #: vector in the norm the Schur factor was taken in.
  vector: NDArray[np.complex128]
  #: ||(A - zI) v|| in that norm, which must equal sigma_min; checks the solve.
  residual: float

  def describe(self) -> dict[str, object]:
    """JSON-safe summary, without the vector itself."""
    rel = (abs(self.residual - self.sigma_min) / self.sigma_min
           if self.sigma_min > 0.0 else 0.0)
    return {
      'z_real': float(self.z.real),
      'z_imag': float(self.z.imag),
      'sigma_min': float(self.sigma_min),
      'residual': float(self.residual),
      'residual_rel_error': float(rel),
      'size': int(self.vector.size),
    }


def pseudomode_at(
  schur_t: NDArray[np.complexfloating],
  schur_z: NDArray[np.complexfloating],
  z: complex,
  tol: float = DEFAULT_MODE_TOL,
  norm: Norm = L2_COEFFICIENTS,
) -> Pseudomode:
  """The pseudomode at `z`, returned in the physical basis.

  `schur_t`, `schur_z` are the halves of `A = Z T Z*`, with `A` the generator in
  `norm`'s coordinates.
  """
  # Z may be tall: n x k for T restricted to a k-dimensional invariant subspace.
  if schur_t.shape[0] != schur_t.shape[1] or schur_z.shape[1] != schur_t.shape[0]:
    raise ValueError(
      f'Schur factor {schur_t.shape} and vectors {schur_z.shape} disagree')

  sigma, mode_t = sigmin_with_mode(z, schur_t, tol=tol)

  # Z is unitary, so ||(T - zI) v_T|| is exactly ||(A - zI) v_A||.
  residual = float(np.linalg.norm(schur_t @ mode_t - z * mode_t))
  vector = np.asarray(norm.to_basis(schur_z @ mode_t), dtype=np.complex128)
  return Pseudomode(
    z=complex(z), sigma_min=sigma, vector=vector, residual=residual)
