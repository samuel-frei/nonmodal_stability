"""The scaled L2 norm: N-weighted resolvent norms, and the maps in and out of it.

A discretised operator is non-normal for two reasons: the basis (FE
coefficients are not orthonormal) and the physics (a non-self-adjoint PDE).
The N-norm removes the first and must leave the second, so the decisive tests
pair a self-adjoint case, which must come out exactly normal, with a
non-self-adjoint one, which must not.
"""

import numpy as np
import pytest
import scipy.linalg
from scipy import sparse

from nonmodal.adapters import read_field_scales
from nonmodal.fields import FIELD_BLOCK_COUNT, FIELD_NAMES, TEMPERATURE_BLOCK_ID
from nonmodal.matrices import assemble_global, assemble_reduced
from nonmodal.norm import (
  L2_COEFFICIENTS,
  SCALED_L2_TAG,
  Norm,
  field_scale_vector,
  norm_matrix,
  scaled_l2_norm,
  weighted_operator,
)
from nonmodal.operator import (
  GENERATOR_CACHE,
  LEGACY_CACHES,
  SCHUR_CACHE,
  SCHURVEC_CACHE,
  load_or_compute_schur,
  load_or_compute_schur_vectors,
)
from nonmodal.pseudomode import pseudomode_at
from nonmodal.pseudospectrum import sample_sigmin

h5py = pytest.importorskip('h5py')


def _graded_mass(n: int = 16) -> np.ndarray:
  """A P1 mass matrix on a strongly graded 1-D mesh: SPD and far from I."""
  h = np.geomspace(1.0, 50.0, n + 1)
  mass = np.zeros((n, n))
  for i in range(n):
    mass[i, i] = (h[i] + h[i + 1]) / 3.0
    if i + 1 < n:
      mass[i, i + 1] = mass[i + 1, i] = h[i + 1] / 6.0
  return mass


def _stiffness(n: int = 16) -> np.ndarray:
  """A symmetric positive-definite stiffness: diffusion, a self-adjoint PDE."""
  return sparse.diags([-1.0, 2.5, -1.0], [-1, 0, 1], shape=(n, n)).toarray()


def _norm_of(mass: np.ndarray) -> Norm:
  return Norm(name='test', description='', cache_tag='t',
              factor=scipy.linalg.cholesky(mass, lower=False))


def _sigmin_and_distance(
  operator: np.ndarray, norm: Norm, tmp_path, points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  schur_t = load_or_compute_schur(operator, str(tmp_path), norm)
  eigvals = np.diag(schur_t)
  distance = np.abs(points[:, None] - eigvals[None, :]).min(axis=1)
  return sample_sigmin(points, schur_t, nprocs=1), distance


POINTS = np.array([0.05 + 0.02j, 0.3 + 0.0j, -0.1 + 0.05j, 0.01 + 0.2j], dtype=np.complex128)


# --- the decisive pair -----------------------------------------------------


def test_a_self_adjoint_operator_is_exactly_normal_in_the_n_norm(tmp_path) -> None:
  """L = -N^-1 S with S symmetric: non-normal in l2 purely through the basis."""
  mass, stiffness = _graded_mass(), _stiffness()
  generator = -np.linalg.solve(mass, stiffness)

  sig_n, dist_n = _sigmin_and_distance(generator, _norm_of(mass), tmp_path / 'n', POINTS)
  sig_2, dist_2 = _sigmin_and_distance(generator, L2_COEFFICIENTS, tmp_path / 'l2', POINTS)

  # Normal: sigma_min is the distance to the spectrum, which is a theorem's equality case.
  np.testing.assert_allclose(sig_n, dist_n, rtol=1e-8)
  # The same operator in l2 shows a strict gap at some point: the basis artefact.
  assert np.max(1.0 - sig_2 / dist_2) > 0.05


def test_a_non_self_adjoint_operator_stays_non_normal_in_the_n_norm(tmp_path) -> None:
  """Advection makes K non-symmetric; no norm of the fields removes that."""
  mass, stiffness = _graded_mass(), _stiffness()
  n = mass.shape[0]
  advection = 0.8 * sparse.diags([-1.0, 1.0], [-1, 1], shape=(n, n)).toarray()
  generator = -np.linalg.solve(mass, stiffness + advection)

  sig_n, dist_n = _sigmin_and_distance(generator, _norm_of(mass), tmp_path, POINTS)

  assert np.all(sig_n <= dist_n * (1.0 + 1e-10))
  assert np.max(1.0 - sig_n / dist_n) > 0.05


# --- the weighted operator and the maps ------------------------------------


def test_weighted_operator_matches_the_dense_definition() -> None:
  rng = np.random.default_rng(11)
  generator = rng.standard_normal((12, 12))
  factor = scipy.linalg.cholesky(_graded_mass(12), lower=False)

  expected = factor @ generator @ np.linalg.inv(factor)
  np.testing.assert_allclose(weighted_operator(generator, factor), expected,
                             rtol=1e-10, atol=1e-10 * np.abs(expected).max())


def test_the_spectrum_does_not_depend_on_the_norm(tmp_path) -> None:
  rng = np.random.default_rng(2)
  generator = rng.standard_normal((14, 14))
  t_n = load_or_compute_schur(generator, str(tmp_path), _norm_of(_graded_mass(14)))
  expected = np.linalg.eigvals(generator)
  # Matched by distance: sorting splits conjugate pairs unpredictably.
  gaps = np.abs(np.diag(t_n)[:, None] - expected[None, :])
  assert gaps.min(axis=1).max() < 1e-10
  assert gaps.min(axis=0).max() < 1e-10


def test_to_basis_inverts_the_factor() -> None:
  rng = np.random.default_rng(4)
  norm = _norm_of(_graded_mass(10))
  assert norm.factor is not None
  x = rng.standard_normal((10, 3)) + 1j * rng.standard_normal((10, 3))
  np.testing.assert_allclose(norm.to_basis(norm.factor @ x), x, rtol=1e-12, atol=1e-12)


def test_l2_is_the_identity_norm() -> None:
  generator = np.arange(9.0).reshape(3, 3)
  vectors = np.ones((3, 2))
  assert L2_COEFFICIENTS.operator(generator) is generator
  assert L2_COEFFICIENTS.to_basis(vectors) is vectors


def test_a_pseudomode_comes_back_in_coefficients_with_its_n_norm_residual(tmp_path) -> None:
  """||(L - z) x||_N = sigma_min and ||x||_N = 1 for the coefficient vector x."""
  mass = _graded_mass()
  n = mass.shape[0]
  advection = 0.8 * sparse.diags([-1.0, 1.0], [-1, 1], shape=(n, n)).toarray()
  generator = -np.linalg.solve(mass, _stiffness() + advection)
  norm = _norm_of(mass)
  assert norm.factor is not None
  schur_t, schur_z = load_or_compute_schur_vectors(generator, str(tmp_path), norm)

  z = 0.05 + 0.02j
  mode = pseudomode_at(schur_t, schur_z, z, norm=norm)
  x = mode.vector

  def n_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(norm.factor @ v))

  assert n_norm(x) == pytest.approx(1.0, rel=1e-10)
  assert n_norm(generator @ x - z * x) == pytest.approx(mode.sigma_min, rel=1e-7)
  # And it is not unit in l2, so the map back really happened.
  assert abs(np.linalg.norm(x) - 1.0) > 1e-3


# --- assembly --------------------------------------------------------------


class _Layout:
  def __init__(self, nrg: int, lg: np.ndarray) -> None:
    self.nrg = self.ncg = nrg
    self.lg = lg


def test_assemble_reduced_matches_scatter_then_mask() -> None:
  """The helper both the dynamics mass and the norm matrix go through."""
  rng = np.random.default_rng(8)
  local = rng.standard_normal((6, 6))
  lg = np.array([0, 2, 2, 3, 1, 4])  # one shared global row, like a shared FE node
  keep = np.array([True, False, True, True, True])

  got = assemble_reduced(sparse.csr_array(local), _Layout(5, lg), keep).toarray()  # type: ignore[arg-type]
  expected = assemble_global(local, 5, 5, lg)[keep][:, keep]
  np.testing.assert_array_equal(got, expected)


def test_norm_matrix_divides_each_block_by_its_scale_squared() -> None:
  nb = 4
  scalar = _graded_mass(nb)
  scales = tuple(float(2 ** k) for k in range(FIELD_BLOCK_COUNT))
  layout = _Layout(FIELD_BLOCK_COUNT * nb, np.arange(FIELD_BLOCK_COUNT * nb))
  keep = np.ones(FIELD_BLOCK_COUNT * nb, dtype=bool)

  matrix = norm_matrix(sparse.csr_array(scalar), scales, layout, keep).toarray()  # type: ignore[arg-type]

  for f, scale in enumerate(scales):
    block = matrix[f * nb:(f + 1) * nb, f * nb:(f + 1) * nb]
    np.testing.assert_allclose(block, scalar / scale**2, rtol=1e-14)
  # Temperature carries its scale only: the dynamics' 1/(gamma-1) is not a norm weight.
  t = TEMPERATURE_BLOCK_ID
  np.testing.assert_allclose(matrix[t * nb, t * nb], scalar[0, 0] / scales[t] ** 2)


def test_norm_matrix_rejects_bad_scales() -> None:
  layout = _Layout(7, np.arange(7))
  keep = np.ones(7, dtype=bool)
  with pytest.raises(ValueError, match='7 field scales'):
    norm_matrix(sparse.eye(1), (1.0,) * 6, layout, keep)  # type: ignore[arg-type]
  with pytest.raises(ValueError, match='positive'):
    norm_matrix(sparse.eye(1), (1.0,) * 6 + (0.0,), layout, keep)  # type: ignore[arg-type]


# --- the deck and an end-to-end export -------------------------------------


DECK = """&harris_sheet_options
  n0=5.196374E16
  den_scale = 2.598187E16
  psi0=0.002045692328575E0,
  t0=1E2,
  eta=742.6
  nu=7424.2
  gamma=1.67
  dt=1.E-7
  lam_b = 0.6075E0
/
"""


def test_field_scales_are_read_from_the_deck(tmp_path) -> None:
  deck = tmp_path / 'oft.in'
  deck.write_text(DECK)
  scales = read_field_scales(str(deck))

  v_a = 0.002045692328575 / np.sqrt(4e-7 * np.pi * 1.672621777e-27 * 5.196374e16)
  assert scales.density == pytest.approx(2.0)
  assert scales.velocity_m_s == pytest.approx(v_a, rel=1e-12)
  assert scales.temperature_eV == pytest.approx(100.0)
  assert scales.flux_T_m == pytest.approx(0.002045692328575 * 0.6075)
  by_name = dict(zip(FIELD_NAMES, field_scale_vector(scales), strict=True))
  assert by_name['U_velx'] == by_name['U_velz'] == scales.velocity_m_s
  assert by_name['U_psi'] == scales.flux_T_m


@pytest.mark.parametrize('key', ['den_scale', 't0', 'lam_b', 'psi0', 'n0'])
def test_a_missing_scale_raises(tmp_path, key: str) -> None:
  deck = tmp_path / 'oft.in'
  deck.write_text('\n'.join(
    line for line in DECK.splitlines() if not line.strip().startswith(f'{key}=')
    and not line.strip().startswith(f'{key} =')) + '\n')
  with pytest.raises(ValueError, match=key):
    read_field_scales(str(deck))


def _write_matrix(f, name: str, matrix: sparse.csr_array, lg: np.ndarray) -> None:
  n = matrix.shape[0]
  f[f'{name}/nr'] = f[f'{name}/nc'] = np.array([n])
  f[f'{name}/nrg'] = f[f'{name}/ncg'] = np.array([lg.max() + 1])
  f[f'{name}/lc'] = matrix.indices + 1
  f[f'{name}/kr'] = matrix.indptr + 1
  f[f'{name}/lg'] = lg + 1
  f[f'{name}/M'] = matrix.data


def test_scaled_l2_norm_of_an_export_factors_the_weighted_mass(tmp_path) -> None:
  nb = 5
  scalar = sparse.csr_array(_graded_mass(nb))
  n = FIELD_BLOCK_COUNT * nb
  with h5py.File(tmp_path / 'lin_ops.h5', 'w') as f:
    _write_matrix(f, 'jacobian', sparse.csr_array(sparse.eye(n)), np.arange(n))
  with h5py.File(tmp_path / 'mass_mat.h5', 'w') as f:
    _write_matrix(f, 'massmat', scalar, np.arange(nb))
  (tmp_path / 'oft.in').write_text(DECK)
  keep = np.ones(n, dtype=bool)
  keep[2 * nb:3 * nb] = False  # drop vely, as the reduction does

  norm = scaled_l2_norm(str(tmp_path / 'lin_ops.h5'), str(tmp_path / 'mass_mat.h5'), keep)

  assert norm.factor is not None
  scales = field_scale_vector(read_field_scales(str(tmp_path / 'oft.in')))
  expected = scipy.linalg.block_diag(*(scalar.toarray() / s**2 for s in scales))[keep][:, keep]
  np.testing.assert_allclose(norm.factor.T @ norm.factor, expected,
                             rtol=1e-10, atol=1e-12 * np.abs(expected).max())
  assert norm.describe()['field_scales']['U_T'] == pytest.approx(100.0)


# --- caches ----------------------------------------------------------------


def test_the_scaled_norm_never_uses_the_l2_or_legacy_cache_names(tmp_path) -> None:
  norm = Norm(name='scaled-L2', description='', cache_tag=SCALED_L2_TAG,
              factor=scipy.linalg.cholesky(_graded_mass(8), lower=False))
  load_or_compute_schur_vectors(np.diag(-np.arange(1.0, 9.0)), str(tmp_path), norm)

  written = {p.name for p in tmp_path.iterdir()}
  assert norm.cache_name(SCHUR_CACHE) == 'generator_schur_l2s.npy'
  assert {norm.cache_name(SCHUR_CACHE), norm.cache_name(SCHURVEC_CACHE)} <= written
  assert not written & ({SCHUR_CACHE, SCHURVEC_CACHE, GENERATOR_CACHE} | set(LEGACY_CACHES))


def test_l2_keeps_the_original_cache_names() -> None:
  assert L2_COEFFICIENTS.cache_name(SCHUR_CACHE) == SCHUR_CACHE
  assert L2_COEFFICIENTS.cache_name(SCHURVEC_CACHE) == SCHURVEC_CACHE


def test_a_cached_l2_factor_is_not_read_as_the_scaled_one(tmp_path) -> None:
  """A cache hit under the l2 name must not satisfy a scaled-norm request."""
  generator = -np.linalg.solve(_graded_mass(8), _stiffness(8))
  np.save(tmp_path / SCHUR_CACHE, np.zeros((8, 8), dtype=np.complex128))
  norm = _norm_of(_graded_mass(8))

  t = load_or_compute_schur(generator, str(tmp_path), norm)

  assert np.abs(np.diag(t)).min() > 0.0
