"""Restrict the analysis to the eigenvalues a finer mesh confirms.

On a finite-element mesh the stiff end of the spectrum belongs to the grid, not
the physics, and it is exactly where the resolvent norm peaks: on the 10x3z Harris
case the Kreiss maximiser sat at the spectral radius and its pseudomode was one
oscillation per element. Pseudospectra of the whole operator are therefore
dominated by modes that do not converge.

The fix is spectral truncation onto an invariant subspace. Reordering the Schur
form so that the selected eigenvalues come first,

    B [Z1 Z2] = [Z1 Z2] [[T11, T12], [0, T22]],

makes `B Z1 = Z1 T11` exact: `T11` *is* `B` restricted to the span of the
selected modes, in the same norm, since `Z1` has orthonormal columns. Its
pseudospectra, Kreiss constant and transient growth are the non-normal
dynamics among those modes, with nothing approximated. What is lost is growth
that needs the discarded modes -- which, when those are unresolved, is the point.

Which eigenvalues to keep is decided by resolution, field by field
(`resolution.py`): a mode is kept when, in every field it occupies, its
effective wavenumber sits below a fraction of the largest one the mesh can
hold. Matching eigenvalues against a finer mesh was tried first and rejected:
the 20x6z spectrum is so dense that a 3-10% shift of the 10x3z spectrum still
found partners for 70-90% of it, about the true match rate. A frequency cutoff
would discard resolved fast waves and keep slow grid-scale diffusion modes.

* `restrict_schur` -- the leading block `T11` (and `Z1`) for a selection.
* `conjugate_closed` -- whether a kept set still pairs up under conjugation.
* `Subspace` -- the record of a restriction, for metadata.
* `restrict_to_resolved` -- restrict a Schur form to the resolved modes.
"""

import os
from dataclasses import dataclass, field
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray
from scipy.linalg.lapack import ztrsen
from scipy.spatial import cKDTree

from .resolution import ModeResolution


def restrict_schur(
  schur_t: NDArray[np.complexfloating],
  schur_z: NDArray[np.complexfloating] | None,
  keep: NDArray[np.bool_],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128] | None]:
  """`T11` and, if `schur_z` is given, `Z1` for the eigenvalues where `keep` is set.

  `B Z1 = Z1 T11` holds to rounding, and `diag(T11)` holds the kept
  eigenvalues (reordered). Uses LAPACK `ztrsen` without condition estimates,
  which would cost a Sylvester solve.
  """
  n = schur_t.shape[0]
  if schur_t.shape != (n, n):
    raise ValueError(f'Schur factor must be square, got {schur_t.shape}')
  if keep.shape != (n,):
    raise ValueError(f'selection has {keep.shape} entries for a {n}x{n} factor')
  count = int(np.count_nonzero(keep))
  if count == 0:
    raise ValueError('the selection keeps no eigenvalues')

  want_vectors = schur_z is not None
  if want_vectors and schur_z.shape != (n, n):  # type: ignore[union-attr]
    raise ValueError(f'Schur vectors {schur_z.shape} do not match T {schur_t.shape}')  # type: ignore[union-attr]
  # ztrsen overwrites both in place, so they are always copied: asfortranarray
  # would hand over the caller's own array when it is already Fortran-ordered.
  # LAPACK never reads Q when it is not wanted, so an untouched buffer suffices
  # (np.empty pages are not committed).
  q = (np.array(schur_z, dtype=np.complex128, order='F', copy=True) if want_vectors
       else np.empty((n, n), dtype=np.complex128, order='F'))
  t = np.array(schur_t, dtype=np.complex128, order='F', copy=True)

  ts, qs, _, m, _, _, info = ztrsen(
    keep.astype(np.int32), t, q, job='N', wantq=int(want_vectors),
    overwrite_t=1, overwrite_q=1)
  if info != 0:
    raise RuntimeError(
      f'ztrsen failed (info={info}): the selected and discarded eigenvalues are too '
      f'close to separate stably')
  if m != count:
    raise RuntimeError(f'ztrsen moved {m} eigenvalues, expected {count}')

  t11 = np.ascontiguousarray(ts[:m, :m])
  z1 = np.ascontiguousarray(qs[:, :m]) if want_vectors else None
  return t11, z1


def _matched(
  eigvals: NDArray[np.complexfloating],
  reference: NDArray[np.complexfloating],
  rtol: float,
) -> NDArray[np.bool_]:
  """Which of `eigvals` have a partner in `reference` within `rtol` relative.

  Matching is one-to-one, closest pairs first, so a cluster cannot all claim
  one reference eigenvalue. Distances are relative to the
  larger modulus of the pair.
  """
  if rtol <= 0.0:
    raise ValueError(f'rtol must be positive, got {rtol}')
  eigvals = np.asarray(eigvals, dtype=np.complex128)
  reference = np.asarray(reference, dtype=np.complex128)
  if reference.size == 0:
    raise ValueError('the reference spectrum is empty')

  tree = cKDTree(np.column_stack([reference.real, reference.imag]))
  pairs: list[tuple[float, int, int]] = []
  # |lam - mu| <= rtol*max(|lam|,|mu|) implies |lam - mu| <= rtol/(1-rtol) * |lam|.
  radius = np.abs(eigvals) * rtol / max(1.0 - rtol, 1e-12)
  hits = tree.query_ball_point(np.column_stack([eigvals.real, eigvals.imag]), radius)
  for i, candidates in enumerate(hits):
    for j in candidates:
      gap = abs(eigvals[i] - reference[j])
      scale = max(abs(eigvals[i]), abs(reference[j]))
      if gap <= rtol * scale:
        pairs.append((gap / scale if scale > 0.0 else 0.0, i, int(j)))

  pairs.sort()
  keep = np.zeros(eigvals.size, dtype=bool)
  used = np.zeros(reference.size, dtype=bool)
  for _, i, j in pairs:
    if not keep[i] and not used[j]:
      keep[i] = used[j] = True
  return keep


#: A kept set is conjugate-closed when its conjugate matches it this closely.
#: Far above the ~1e-9 roundoff a complex Schur form leaves on real pairs.
CONJUGATE_RTOL = 1e-6


def conjugate_closed(eigvals: NDArray[np.complexfloating]) -> bool:
  """Whether `eigvals` equals its own conjugate as a multiset, to `CONJUGATE_RTOL`.

  A real operator's spectrum is; a selection can break that by keeping one of
  a pair. Half-plane sampling and the Kreiss real-axis argument both rely on it,
  so callers check rather than assume. A numerically real eigenvalue is its own
  partner, so no judgement about which roundoff counts as zero is needed.
  """
  eigvals = np.asarray(eigvals, dtype=np.complex128)
  return bool(_matched(eigvals, np.conj(eigvals), CONJUGATE_RTOL).all())


@dataclass(frozen=True)
class Subspace:
  """How the analysed operator was restricted, for run metadata."""

  #: False when the whole operator was analysed.
  restricted: bool
  dimension: int
  full_dimension: int
  #: What decided the kept set; empty when nothing was restricted.
  rule: str = ''
  #: Largest k_f / k_max a kept mode may have, in its worst field.
  cutoff: float = 0.0
  #: The mesh's largest wavenumber per field, 1/m.
  k_max: dict[str, float] = field(default_factory=dict)
  stiffmat: str = ''
  #: The kept eigenvalues' largest modulus, which bounds the timescales analysed.
  max_abs_eigenvalue: float = 0.0
  #: False if the kept set split a conjugate pair; mirroring is then unsafe.
  conjugate_closed: bool = True
  kept_eigenvalues: tuple[complex, ...] = field(default=(), repr=False)

  def describe(self) -> dict[str, Any]:
    """A JSON-ready dictionary."""
    return {
      'restricted': self.restricted,
      'dimension': self.dimension,
      'full_dimension': self.full_dimension,
      'rule': self.rule,
      'cutoff': self.cutoff,
      'k_max': dict(self.k_max),
      'stiffmat': self.stiffmat,
      'max_abs_eigenvalue': self.max_abs_eigenvalue,
      'conjugate_closed': self.conjugate_closed,
      'kept_eigenvalues': [[z.real, z.imag] for z in self.kept_eigenvalues],
    }


#: Recorded in metadata, so a reader knows what `cutoff` measures.
RESOLUTION_RULE = 'per-field: max_f k_f/k_max < cutoff over the fields a mode occupies'


def whole_operator(schur_t: NDArray[np.complexfloating]) -> Subspace:
  """The record for an unrestricted analysis."""
  n = int(schur_t.shape[0])
  return Subspace(restricted=False, dimension=n, full_dimension=n)


@overload
def restrict_to_resolved(
  schur_t: NDArray[np.complexfloating], schur_z: NDArray[np.complexfloating],
  resolution: ModeResolution, cutoff: float, stiffmat: str,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], Subspace]: ...
@overload
def restrict_to_resolved(
  schur_t: NDArray[np.complexfloating], schur_z: None,
  resolution: ModeResolution, cutoff: float, stiffmat: str,
) -> tuple[NDArray[np.complex128], None, Subspace]: ...
def restrict_to_resolved(
  schur_t: NDArray[np.complexfloating],
  schur_z: NDArray[np.complexfloating] | None,
  resolution: ModeResolution,
  cutoff: float,
  stiffmat: str,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128] | None, Subspace]:
  """Restrict a Schur form to the modes `resolution` finds resolved at `cutoff`."""
  eigvals = np.asarray(np.diag(schur_t), dtype=np.complex128)
  n = eigvals.size
  if not np.array_equal(resolution.eigenvalues, eigvals):
    raise ValueError('the mode resolution was computed for a different Schur factor')
  keep = resolution.resolved(cutoff)
  if not keep.any():
    raise ValueError(
      f'no mode is resolved at cutoff {cutoff:g}: the best has max_f k_f/k_max = '
      f'{resolution.worst().min():.3g}')
  kept = eigvals[keep]
  closed = conjugate_closed(kept)
  slowest = eigvals[np.argmax(eigvals.real)]
  print(f'subspace: kept {kept.size} of {n} modes with max_f k_f/k_max < {cutoff:g}; '
        f'largest kept |lambda| = {np.abs(kept).max():.4g} /s (full spectrum to '
        f'{np.abs(eigvals).max():.4g} /s)', flush=True)
  if not keep[np.argmax(eigvals.real)]:
    print(f'note: the rightmost eigenvalue {slowest:.4g} /s is not resolved '
          f'(max_f k_f/k_max = {resolution.worst()[np.argmax(eigvals.real)]:.3g}) '
          f'and is excluded', flush=True)
  if not closed:
    print('warning: the kept eigenvalues are not closed under conjugation; '
          'half-plane mirroring is disabled', flush=True)

  t11, z1 = restrict_schur(schur_t, schur_z, keep)
  return t11, z1, Subspace(
    restricted=True, dimension=int(kept.size), full_dimension=n,
    rule=RESOLUTION_RULE, cutoff=float(cutoff), k_max=resolution.k_max_by_field(),
    stiffmat=os.path.abspath(stiffmat),
    max_abs_eigenvalue=float(np.abs(kept).max()), conjugate_closed=closed,
    kept_eigenvalues=tuple(complex(z) for z in kept))
