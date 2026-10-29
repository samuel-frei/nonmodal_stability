"""Which eigenmodes the mesh resolves, judged field by field.

* `check_laplacian` -- reject a matrix that is not a Lagrange Laplacian.
* `reduced_stiffness` -- the stiffness `blockdiag(K_s)` on the reduced rows.
* `triangular_eigenvectors` -- every eigenvector of a triangular `T` (LAPACK ztrevc).
* `ModeResolution` -- per-mode, per-field wavenumbers, and the resolved selection.
* `mode_resolution` -- compute it from the Schur form.
* `load_or_compute_resolution` -- the same, cached beside the Schur factor.

OFT's linear Harris runs write the scalar Lagrange Laplacian
`K_ij = int grad(phi_i) . grad(phi_j) dA` to `stiff_mat.h5` beside the mass
matrix, with the same Dirichlet boundary rows replaced by identity rows. Every
field shares that one scalar basis, so one `K` serves all of them. With the
scalar mass `M`, field `f` of a mode `x` has effective wavenumber
`k_f^2 = x_f K x_f / x_f M x_f`, and the mesh can hold at most
`k_max^2 = max eig(M^-1 K)`.

A mode is resolved when `k_f / k_max < cutoff` in every field it occupies:
the worst field decides. Weighting by the norm instead would hide a field
that carries little of the mode's energy but is grid-scale -- the slow
pseudomode on 10x3z is 0.1% velocity, and that velocity sat at 0.2 k_max while
the norm-weighted wavenumber looked smooth. A field is skipped only when it
is absent to rounding (`ZERO_FIELD_SHARE`), never because it is small.

Eigenvectors of a nearly degenerate cluster are ill-determined, so their
per-mode wavenumbers depend on the basis LAPACK happens to pick. Conjugate
pairs of a real operator have equal wavenumbers in exact arithmetic; one
that straddles the cutoff in rounding leaves the kept set not
conjugate-closed, which `subspace` detects.
"""

import ctypes
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import cython_lapack

from .fields import FIELD_BLOCK_COUNT, FIELD_NAMES
from .matrices import HDF5Matrix, assemble_reduced
from .norm import Norm, norm_matrix

#: HDF5 group the export writes the stiffness matrix under.
STIFFNESS_GROUP = '/stiffmat'
#: Largest row sum allowed on an interior row, relative to the row's absolute sum.
ROW_SUM_TOLERANCE = 1e-10
#: Largest asymmetry allowed in the reduced matrix, relative to its largest entry.
SYMMETRY_TOLERANCE = 1e-12


def check_laplacian(scalar: Any) -> None:
  """Raise unless every row is a Dirichlet identity row or sums to zero.

  A Lagrange basis sums to one everywhere, so the Laplacian annihilates
  constants and every unconstrained row sums to zero. A mass matrix, whose
  rows sum to the area each basis function covers, fails this.
  """
  csr = sparse.csr_array(scalar)
  row_sum = np.abs(np.asarray(csr.sum(axis=1)).ravel())
  row_abs = np.asarray(abs(csr).sum(axis=1)).ravel()
  diagonal = csr.diagonal()
  dirichlet = np.isclose(row_abs, 1.0) & np.isclose(diagonal, 1.0)
  bad = ~dirichlet & (row_sum > ROW_SUM_TOLERANCE * row_abs)
  if np.any(bad):
    worst = int(np.argmax(np.where(bad, row_sum / np.maximum(row_abs, 1e-300), 0.0)))
    raise ValueError(
      f'stiffness matrix is not a Laplacian: {int(np.count_nonzero(bad))} rows do not '
      f'sum to zero (worst row {worst}: sum {row_sum[worst]:.3e} of {row_abs[worst]:.3e})')


def reduced_stiffness(
  jacobian_path: str,
  stiffmat_path: str,
  keep_global: Any,
) -> Any:
  """The reduced stiffness `blockdiag(K_s)`, sparse, on the generator's rows.

  Assembled and reduced through the same path as the dynamics mass and the
  norm matrix, so field `f`'s rows of the result are `K` restricted to the
  nodes that field keeps. The result must be symmetric: a kept row that the
  export constrained (an identity row whose column is not) means the
  reduction's boundary rows and the export's disagree, and this raises.
  """
  layout = HDF5Matrix(jacobian_path, '/jacobian')
  stiffness = HDF5Matrix(stiffmat_path, STIFFNESS_GROUP)
  if FIELD_BLOCK_COUNT * stiffness.nr != layout.nr:
    raise ValueError(
      f'stiffness matrix has {stiffness.nr} rows, but the Jacobian has {layout.nr} = '
      f'{FIELD_BLOCK_COUNT} x {layout.nr / FIELD_BLOCK_COUNT:g}')
  check_laplacian(stiffness.csr_rep)

  blocks = [stiffness.csr_rep] * FIELD_BLOCK_COUNT
  reduced = assemble_reduced(sparse.block_diag(blocks, format='csr'), layout, keep_global)
  scale = float(abs(reduced).max())
  asymmetry = float(abs(reduced - reduced.T).max()) if reduced.nnz else 0.0
  if asymmetry > SYMMETRY_TOLERANCE * scale:
    raise ValueError(
      f'reduced stiffness matrix is not symmetric (|K - K^T| = {asymmetry:.3e} of '
      f'{scale:.3e}): the reduction keeps rows the export constrained')
  return reduced


def reduced_scalar_mass(
  jacobian_path: str,
  massmat_path: str,
  keep_global: Any,
) -> Any:
  """The reduced `blockdiag(M_s)`, with no per-field factor, on the generator's rows."""
  layout = HDF5Matrix(jacobian_path, '/jacobian')
  mass = HDF5Matrix(massmat_path, '/massmat')
  return norm_matrix(mass.csr_rep, (1.0,) * FIELD_BLOCK_COUNT, layout, keep_global)


# --- eigenvectors of a triangular matrix --------------------------------------

def _ztrevc() -> Any:
  """LAPACK ztrevc, reached through scipy's Cython LAPACK, which exports it.

  `scipy.linalg.lapack` has no wrapper for it, but `cython_lapack` publishes
  every routine as a capsule holding the C function pointer, with 32-bit ints.
  """
  capsule = cython_lapack.__pyx_capi__['ztrevc']
  get_name = ctypes.pythonapi.PyCapsule_GetName
  get_name.restype = ctypes.c_char_p
  get_name.argtypes = [ctypes.py_object]
  get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
  get_pointer.restype = ctypes.c_void_p
  get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
  address = get_pointer(capsule, get_name(capsule))
  int_p, ptr = ctypes.POINTER(ctypes.c_int), ctypes.c_void_p
  prototype = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_char_p, ptr, int_p, ptr, int_p, ptr, int_p,
    ptr, int_p, int_p, int_p, ptr, ptr, int_p)
  return prototype(address)


def triangular_eigenvectors(schur_t: NDArray[np.complexfloating]) -> NDArray[np.complex128]:
  """Right eigenvectors of upper-triangular `T`, column `i` for `T[i, i]`.

  The result is upper triangular and Fortran-ordered; each column is scaled so
  its largest entry has |re| + |im| = 1. Repeated eigenvalues are perturbed as
  LAPACK does, so a defective cluster gets nearly parallel vectors.
  """
  n = int(schur_t.shape[0])
  if schur_t.shape != (n, n):
    raise ValueError(f'T must be square, got {schur_t.shape}')
  t = np.array(schur_t, dtype=np.complex128, order='F', copy=True)
  vr = np.zeros((n, n), dtype=np.complex128, order='F')
  work = np.empty(2 * n, dtype=np.complex128)
  rwork = np.empty(n, dtype=np.float64)
  select = np.zeros(1, dtype=np.int32)  # not referenced for howmny='A'
  c_n, c_m, info = ctypes.c_int(n), ctypes.c_int(0), ctypes.c_int(0)
  lead = ctypes.c_int(max(n, 1))
  _ztrevc()(
    b'R', b'A', select.ctypes.data, ctypes.byref(c_n), t.ctypes.data, ctypes.byref(lead),
    None, ctypes.byref(ctypes.c_int(1)), vr.ctypes.data, ctypes.byref(lead),
    ctypes.byref(c_n), ctypes.byref(c_m), work.ctypes.data, rwork.ctypes.data,
    ctypes.byref(info))
  if info.value != 0:
    raise RuntimeError(f'ztrevc failed (info={info.value})')
  if c_m.value != n:
    raise RuntimeError(f'ztrevc returned {c_m.value} vectors, expected {n}')
  return vr


# --- per-mode resolution ------------------------------------------------------

#: The worst field must sit below this fraction of the mesh's largest wavenumber.
#: 0.2 left the 10x3z velocity unconverged; 0.02-0.06 converged.
DEFAULT_RESOLUTION_CUTOFF = 0.1
#: A field carrying less than this share of a mode's scaled-L2 energy is
#: absent to rounding (amplitude ~1e-10) and does not vote.
ZERO_FIELD_SHARE = 1e-20
#: Eigenvectors are formed this many at a time, to bound memory.
EIGENVECTOR_CHUNK = 512
#: Cache of the per-mode resolution, tagged with the norm like the Schur factor.
RESOLUTION_CACHE = 'mode_resolution.npz'


@dataclass(frozen=True)
class ModeResolution:
  """Per-mode, per-field wavenumbers of every eigenvector of `T`.

  Row `i` is the eigenvalue `T[i, i]`; column `j` is block `fields[j]`.
  """

  eigenvalues: NDArray[np.complex128]
  fields: tuple[int, ...]
  #: k_f in 1/m.
  wavenumber: NDArray[np.float64]
  #: Field f's share of the mode's scaled-L2 energy.
  share: NDArray[np.float64]
  #: Largest wavenumber each field's nodes can hold, 1/m.
  k_max: NDArray[np.float64]

  def ratio(self) -> NDArray[np.float64]:
    """`k_f / k_max`, with fields absent from a mode set to 0."""
    ratio = self.wavenumber / self.k_max[np.newaxis, :]
    return np.where(self.share >= ZERO_FIELD_SHARE, ratio, 0.0)

  def worst(self) -> NDArray[np.float64]:
    """The largest `k_f / k_max` over the fields each mode occupies."""
    return self.ratio().max(axis=1)

  def resolved(self, cutoff: float) -> NDArray[np.bool_]:
    """Modes whose worst field is below `cutoff`."""
    return self.worst() < cutoff

  def k_max_by_field(self) -> dict[str, float]:
    """JSON-ready `k_max` per field name."""
    return {FIELD_NAMES[b]: float(k) for b, k in zip(self.fields, self.k_max, strict=True)}


def _largest_wavenumber(stiffness: Any, mass: Any) -> float:
  """`sqrt(max eig(M^-1 K))` for one field's block."""
  n = stiffness.shape[0]
  # A fixed start vector, as for the sampler: ARPACK's random one is irreproducible.
  values = scipy.sparse.linalg.eigsh(
    sparse.csc_array(stiffness), k=1, M=sparse.csc_array(mass), which='LA',
    v0=np.ones(n), tol=1e-10, return_eigenvectors=False)
  return float(np.sqrt(values.max()))


def mode_resolution(
  schur_t: NDArray[np.complexfloating],
  schur_z: NDArray[np.complexfloating],
  norm: Norm,
  stiffness: Any,
  mass: Any,
  row_fields: NDArray[np.integer],
) -> ModeResolution:
  """Wavenumbers of every eigenvector of `B = Z T Z*`, field by field.

  `stiffness` and `mass` are the reduced block-diagonal scalar matrices and
  `row_fields` names each reduced row's block. Eigenvectors of `T` come from
  ztrevc; `R^-1 Z v` takes each back to finite-element coefficients.
  """
  n = int(schur_t.shape[0])
  if schur_z.shape != (stiffness.shape[0], n) or row_fields.shape != (stiffness.shape[0],):
    raise ValueError(
      f'Schur vectors {schur_z.shape}, stiffness {stiffness.shape} and row fields '
      f'{row_fields.shape} do not line up')
  fields = tuple(int(b) for b in np.unique(row_fields))
  rows = [np.flatnonzero(row_fields == b) for b in fields]
  k_blocks = [sparse.csr_array(stiffness[r][:, r]) for r in rows]
  m_blocks = [sparse.csr_array(mass[r][:, r]) for r in rows]
  k_max = np.array([_largest_wavenumber(k, m) for k, m in zip(k_blocks, m_blocks, strict=True)])
  scales = np.array([norm.field_scales.get(FIELD_NAMES[b], 1.0) for b in fields])
  print('mode resolution: k_max = ' + ', '.join(
    f'{FIELD_NAMES[b]} {k:.4g}' for b, k in zip(fields, k_max, strict=True)) + ' /m', flush=True)

  vectors = triangular_eigenvectors(schur_t)
  wavenumber = np.empty((n, len(fields)))
  energy = np.empty((n, len(fields)))
  for start in range(0, n, EIGENVECTOR_CHUNK):
    stop = min(start + EIGENVECTOR_CHUNK, n)
    # Columns start..stop-1 are zero below row stop-1: T's vectors are triangular.
    coefficients = norm.to_basis(schur_z[:, :stop] @ vectors[:stop, start:stop])
    for j, (r, k_f, m_f) in enumerate(zip(rows, k_blocks, m_blocks, strict=True)):
      x = coefficients[r]
      kinetic = np.real(np.sum(x.conj() * (k_f @ x), axis=0))
      weight = np.real(np.sum(x.conj() * (m_f @ x), axis=0))
      wavenumber[start:stop, j] = np.sqrt(np.maximum(kinetic, 0.0) / np.maximum(weight, 1e-300))
      energy[start:stop, j] = weight / scales[j] ** 2
  del vectors
  share = energy / energy.sum(axis=1, keepdims=True)
  return ModeResolution(
    eigenvalues=np.asarray(np.diag(schur_t), dtype=np.complex128).copy(),
    fields=fields, wavenumber=wavenumber, share=share, k_max=k_max)


def load_or_compute_resolution(
  schur_t: NDArray[np.complexfloating],
  schur_z: NDArray[np.complexfloating] | None,
  norm: Norm,
  cache_dir: str,
  jacobian_path: str,
  massmat_path: str,
  stiffmat_path: str,
  keep_global: NDArray[np.bool_],
) -> ModeResolution | None:
  """The cached resolution if it belongs to this `T`, else compute it.

  A cache counts only if its eigenvalues equal `diag(T)` exactly, which ties
  it to one Schur factor. With no cache and no `schur_z`, returns None: the
  caller must load `Z` and ask again.
  """
  path = os.path.abspath(os.path.join(cache_dir, norm.cache_name(RESOLUTION_CACHE)))
  diagonal = np.asarray(np.diag(schur_t), dtype=np.complex128)
  if os.path.isfile(path):
    with np.load(path) as data:
      cached = ModeResolution(
        eigenvalues=data['eigenvalues'], fields=tuple(int(b) for b in data['fields']),
        wavenumber=data['wavenumber'], share=data['share'], k_max=data['k_max'])
    if np.array_equal(cached.eigenvalues, diagonal):
      print(f'loaded cached mode resolution: {path}', flush=True)
      return cached
    print(f'ignoring {path}: its eigenvalues are not this Schur factor\'s', flush=True)
  if schur_z is None:
    return None

  stiffness = reduced_stiffness(jacobian_path, stiffmat_path, keep_global)
  mass = reduced_scalar_mass(jacobian_path, massmat_path, keep_global)
  nrg = keep_global.shape[0]
  row_fields = np.repeat(np.arange(FIELD_BLOCK_COUNT), nrg // FIELD_BLOCK_COUNT)[keep_global]
  result = mode_resolution(schur_t, schur_z, norm, stiffness, mass, row_fields)
  os.makedirs(cache_dir, exist_ok=True)
  np.savez(path, eigenvalues=result.eigenvalues, fields=np.array(result.fields),
           wavenumber=result.wavenumber, share=result.share, k_max=result.k_max)
  print(f'cached mode resolution: {path}', flush=True)
  return result
