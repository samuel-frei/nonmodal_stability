"""Pseudomodes: the vector a near-singular resolvent actually amplifies.

For a point `z`, the pseudomode is the right singular vector `v` of `zI - A`
belonging to `sigma_min`. It satisfies

    ||(A - zI) v|| = sigma_min(zI - A)

so it is an approximate eigenvector carrying a known residual. On a strongly
non-normal operator `sigma_min` is tiny far from any eigenvalue, and `v` is then
a direction the operator very nearly leaves invariant even though `z` is nowhere
near the spectrum.

This runs the same inverse iteration the sampler does, at the requested point:
`sigma_min` and its eigenvector come out of one `eigsh` call on
`(zI - T)^-1 (zI - T)^-H = V Sigma^-2 V*`, and that eigenvector *is* the
pseudomode. Sampling discarded it as `vals, _`; `sigmin_with_mode` keeps it.

So the vector is free relative to the value -- ARPACK produces both -- but the
solve itself is not free. Sampled vectors are not stored, so a mode costs a
fresh iteration (~110 matvecs at the default tolerance, two triangular solves
each) even at a point the run already visited.

The genuinely new step is the change of basis. `sigma_min` is invariant under
`A = Z T Z*`, but singular vectors are not, so the Schur vectors carry the
result back:

    v_A = Z v_T

`Z` is produced by the same `scipy.linalg.schur` call that produces `T` and is
cached alongside it (`operator.SCHURVEC_CACHE`).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .pseudospectrum import DEFAULT_MODE_TOL, sigmin_with_mode


@dataclass(frozen=True)
class Pseudomode:
  """One pseudomode, with the evidence that it is one."""

  z: complex
  sigma_min: float
  #: Unit vector in the reduced physical basis that `keep_global` indexes.
  vector: NDArray[np.complex128]
  #: ||(A - zI) v||, which must equal sigma_min. Evaluated from the operator
  #: rather than the inverse iteration, so it checks the solve instead of
  #: restating it.
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
) -> Pseudomode:
  """The pseudomode at `z`, returned in the physical basis.

  `schur_t` and `schur_z` are the two halves of `A = Z T Z*`, as produced by
  `operator.load_or_compute_schur_vectors`.
  """
  if schur_t.shape != schur_z.shape:
    raise ValueError(
      f'Schur factor {schur_t.shape} and vectors {schur_z.shape} disagree')

  sigma, mode_t = sigmin_with_mode(z, schur_t, tol=tol)

  # Residual in the Schur basis. Z is unitary, so ||(T - zI) v_T|| is exactly
  # ||(A - zI) v_A|| -- checked without touching the n-by-n physical operator.
  residual = float(np.linalg.norm(schur_t @ mode_t - z * mode_t))
  vector = np.asarray(schur_z @ mode_t, dtype=np.complex128)
  return Pseudomode(
    z=complex(z), sigma_min=sigma, vector=vector, residual=residual)
