"""The norm every resolvent norm, Kreiss constant and pseudomode is measured in.

The raw state vector holds finite-element coefficients in each field's own
units. Its Euclidean norm (l2) depends on the mesh, since the basis is not
orthonormal, and on the units, since psi in T m is ~1e-6 of a velocity in m/s.
Neither converges with resolution, so it cannot support a parameter scan.

The scaled L2 norm fixes both. With an SPD norm matrix `N = R^T R` (upper
Cholesky `R`),

    ||x||_N = ||R x||_2,   y = R x,   dy/dt = B y,   B = R L R^-1,

so everything downstream -- Schur, sampling, Kreiss -- runs unchanged on `B`,
and a vector found in `y` returns to coefficients as `x = R^-1 y`. Here
`N = blockdiag(M_s / s_f^2)`: the scalar mass matrix `M_s` in every field, each
divided by the square of that field's typical size `s_f` from the deck. So
`||x||_N^2 = sum_f int (u_f / s_f)^2 dA`, a dimensionless L2 norm of the fields.

The temperature block carries no `1/(gamma-1)` here. That factor belongs to
the dynamics mass, where OFT's equations put it; this is the plain L2 norm.

* `Norm` -- a norm: its name, its factor `R`, and the maps into and out of it.
* `L2_COEFFICIENTS` -- the Euclidean norm of the coefficients, `R = I`.
* `field_scale_vector` -- `s_f` for each of the seven field blocks.
* `norm_matrix` -- the reduced `N`, assembled exactly like the dynamics mass.
* `weighted_operator` -- `B = R L R^-1` without forming an inverse.
* `scaled_l2_norm` -- the scaled L2 norm of an export, read from its deck.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from scipy import sparse

from .adapters import FieldScales, find_input_deck, read_field_scales
from .fields import FIELD_BLOCK_COUNT, FIELD_NAMES
from .matrices import HDF5Matrix, assemble_reduced

#: Cache-name tag of the scaled L2 norm's Schur factors. The l2 norm's tag is
#: empty, so it keeps the original `generator_schur*.npy` names.
SCALED_L2_TAG = 'l2s'


@dataclass(frozen=True)
class Norm:
  """A norm `||x|| = ||R x||_2`; `R is None` means the plain Euclidean norm."""

  name: str
  description: str
  #: Distinguishes this norm's cached factorisations; '' for l2.
  cache_tag: str
  #: Upper-triangular Cholesky factor of the norm matrix, or None for R = I.
  factor: NDArray[np.float64] | None = field(default=None, repr=False)
  #: s_f per field name, for the record; empty for l2.
  field_scales: dict[str, float] = field(default_factory=dict)

  def operator(self, generator: NDArray[np.float64]) -> NDArray[np.float64]:
    """The generator as it acts in this norm's coordinates."""
    if self.factor is None:
      return generator
    return weighted_operator(generator, self.factor)

  def to_basis(self, vectors: NDArray[Any]) -> NDArray[Any]:
    """Map vectors (columns) from this norm's coordinates back to coefficients."""
    if self.factor is None:
      return vectors
    return np.asarray(scipy.linalg.solve_triangular(self.factor, vectors, lower=False))

  def cache_name(self, filename: str) -> str:
    """`filename` tagged with this norm, so factorisations never share a name."""
    if not self.cache_tag:
      return filename
    stem, dot, ext = filename.rpartition('.')
    return f'{stem}_{self.cache_tag}{dot}{ext}'

  def describe(self) -> dict[str, Any]:
    """A JSON-ready record; plots in different norms look alike."""
    return {
      'name': self.name,
      'description': self.description,
      'field_scales': dict(self.field_scales),
    }


L2_COEFFICIENTS = Norm(
  name='l2',
  description='Euclidean norm of the reduced finite-element coefficients '
              '(mesh- and unit-dependent)',
  cache_tag='',
)


def field_scale_vector(scales: FieldScales) -> tuple[float, ...]:
  """`s_f` for each field block, in `FIELD_NAMES` order."""
  by_name = {
    'U_n': scales.density,
    'U_velx': scales.velocity_m_s,
    'U_vely': scales.velocity_m_s,
    'U_velz': scales.velocity_m_s,
    'U_T': scales.temperature_eV,
    'U_psi': scales.flux_T_m,
    'U_by': scales.field_T,
  }
  return tuple(by_name[name] for name in FIELD_NAMES)


def norm_matrix(
  scalar_mass: Any,
  field_scales: tuple[float, ...],
  layout: HDF5Matrix,
  keep_global: NDArray[np.bool_],
) -> Any:
  """The reduced norm matrix `N = blockdiag(M_s / s_f^2)`, sparse.

  Assembled and reduced through the same path as the dynamics mass, so its
  rows line up with the reduced generator's.
  """
  if len(field_scales) != FIELD_BLOCK_COUNT:
    raise ValueError(f'need {FIELD_BLOCK_COUNT} field scales, got {len(field_scales)}')
  if min(field_scales) <= 0.0:
    raise ValueError(f'field scales must be positive, got {field_scales}')
  blocks = [scalar_mass / (scale * scale) for scale in field_scales]
  return assemble_reduced(sparse.block_diag(blocks, format='csr'), layout, keep_global)


def weighted_operator(
  generator: NDArray[np.float64],
  factor: NDArray[np.float64],
) -> NDArray[np.float64]:
  """`B = R L R^-1` for upper-triangular `R`, by a triangular solve.

  `B R = R L`, so `R^T B^T = (R L)^T`, which is one solve against `R^T`.
  """
  transposed = scipy.linalg.solve_triangular(
    factor, (factor @ generator).T, trans='T', lower=False)
  return np.ascontiguousarray(np.asarray(transposed, dtype=np.float64).T)


def scaled_l2_norm(
  jacobian_path: str,
  massmat_path: str,
  keep_global: NDArray[np.bool_],
) -> Norm:
  """The scaled L2 norm of an export, with field scales read from its deck.

  Cheap next to the factorisation it feeds, so it is rebuilt rather than cached.
  """
  deck = find_input_deck(jacobian_path)
  scales = read_field_scales(deck)
  per_field = field_scale_vector(scales)
  print('scaled L2 norm, field scales: ' + ', '.join(
    f'{name}={scale:.4g}' for name, scale in zip(FIELD_NAMES, per_field, strict=True)),
    flush=True)

  layout = HDF5Matrix(jacobian_path, '/jacobian')
  mass = HDF5Matrix(massmat_path, '/massmat')
  matrix = norm_matrix(mass.csr_rep, per_field, layout, keep_global)
  del layout, mass
  factor = np.asarray(scipy.linalg.cholesky(matrix.toarray(), lower=False), dtype=np.float64)

  return Norm(
    name='scaled-L2',
    description='L2 norm of the fields, each divided by its typical size: '
                'n0/den_scale, v_A at n0, t0, psi0*lam_b, psi0',
    cache_tag=SCALED_L2_TAG,
    factor=factor,
    field_scales=dict(zip(FIELD_NAMES, per_field, strict=True)),
  )
