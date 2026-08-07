"""Pseudomodes: the vector a near-singular resolvent actually amplifies.

For a point `z`, the pseudomode is the right singular vector `v` of `zI - A`
belonging to `sigma_min`. It satisfies

    ||(A - zI) v|| = sigma_min(zI - A)

so it is an approximate eigenvector carrying a known residual. On a strongly
non-normal operator `sigma_min` is tiny far from any eigenvalue, and `v` is then
a direction the operator very nearly leaves invariant even though `z` is nowhere
near the spectrum.

Nothing here re-solves anything. Sampling already runs inverse iteration on
`(zI - T)^-1 (zI - T)^-H = V Sigma^-2 V*` at every point and discards the
eigenvector, which *is* the pseudomode -- `sigmin_with_mode` keeps it. The only
extra step is the change of basis: `sigma_min` is invariant under `A = Z T Z*`,
but singular vectors are not, so the Schur vectors carry the result back:

    v_A = Z v_T

`Z` is produced by the same `scipy.linalg.schur` call that produces `T` and is
now cached alongside it (`operator.SCHURVEC_CACHE`).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .pseudospectrum import DEFAULT_MODE_TOL, sigmin_with_mode

#: Ways to pick a point out of a finished run's samples.
SAMPLE_RULES: tuple[str, ...] = ('rightmost', 'min-sigmin', 'kreiss')


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


def _on_sampled_boundary(point: complex, z: NDArray[np.complex128]) -> bool:
  """Whether `point` lies on an edge of the sampled region.

  Compared directly against the sample extrema rather than through `Bounds`,
  whose strict-ordering check would reject a degenerate (single row or column)
  sample set that is otherwise perfectly answerable.
  """
  return bool(
    point.real in (float(z.real.min()), float(z.real.max()))
    or point.imag in (float(z.imag.min()), float(z.imag.max())))


def select_from_samples(
  z: NDArray[np.complex128],
  sigmin: NDArray[np.float64],
  rule: str,
) -> tuple[complex, bool]:
  """Pick a sample point by `rule`; also report whether it sits on the edge.

  The boundary flag is not incidental. The sampled region is derived from the
  spectrum, so the rightmost sample is the right edge of that box -- set by
  where sampling stopped, not by the operator -- and the pseudospectrum
  generally continues past it. Callers must say so rather than present such a
  point as though it were the true extremum.

  Rules:

  * `min-sigmin` -- the most nearly singular sample. Parameter-free and
    unambiguous, but usually sits on top of an eigenvalue.
  * `rightmost` -- largest `Re z`, breaking ties by smallest `sigma_min`.
    Almost always on the boundary; read the warning.
  * `kreiss` -- largest `Re z / sigma_min` over the right half-plane, i.e. the
    Kreiss ratio evaluated on the samples. This is the point contributing most
    to the transient-growth lower bound. Evaluated, not optimised: the true
    maximiser generally lies outside the sampled box.
  """
  z = np.asarray(z).ravel()
  sigmin = np.asarray(sigmin).ravel()
  if z.size == 0:
    raise ValueError('no samples to choose a point from')
  if z.shape != sigmin.shape:
    raise ValueError(f'samples disagree: {z.shape} points vs {sigmin.shape} values')

  if rule == 'min-sigmin':
    index = int(np.argmin(sigmin))
  elif rule == 'rightmost':
    at_edge = np.flatnonzero(z.real == z.real.max())
    index = int(at_edge[np.argmin(sigmin[at_edge])])
  elif rule == 'kreiss':
    right = z.real > 0.0
    if not right.any():
      raise ValueError(
        'no sampled point has Re z > 0, so the Kreiss ratio is undefined here; '
        'the sampled region never reached the right half-plane')
    ratio = np.where(
      right, z.real / np.maximum(sigmin, np.finfo(float).tiny), -np.inf)
    index = int(np.argmax(ratio))
  else:
    raise ValueError(
      f'unknown from-samples rule {rule!r}; choose one of {", ".join(SAMPLE_RULES)}')

  chosen = complex(z[index])
  return chosen, _on_sampled_boundary(chosen, z)
