"""The exported stiffness matrix, and per-field resolution of eigenmodes."""

import h5py
import numpy as np
import pytest
import scipy.linalg
from scipy import sparse

from nonmodal.fields import FIELD_BLOCK_COUNT, FIELD_NAMES
from nonmodal.norm import L2_COEFFICIENTS, Norm
from nonmodal.resolution import (
  ZERO_FIELD_SHARE,
  ModeResolution,
  check_laplacian,
  load_or_compute_resolution,
  mode_resolution,
  reduced_stiffness,
  triangular_eigenvectors,
)


def _laplacian_1d(nb: int) -> np.ndarray:
  """P1 Laplacian on a graded 1-D mesh, end rows replaced by identity rows."""
  nodes = np.cumsum(np.linspace(1.0, 2.0, nb)) - 1.0
  stiffness = np.zeros((nb, nb))
  for i, h in enumerate(np.diff(nodes)):
    stiffness[i:i + 2, i:i + 2] += np.array([[1.0, -1.0], [-1.0, 1.0]]) / h
  for end in (0, nb - 1):
    stiffness[end] = 0.0
    stiffness[end, end] = 1.0
  return stiffness


def _mass_1d(nb: int) -> np.ndarray:
  return (np.diag(np.full(nb, 4.0)) + np.diag(np.ones(nb - 1), 1)
          + np.diag(np.ones(nb - 1), -1)) / 6.0


def _write_matrix(f, name: str, matrix: sparse.csr_array, lg: np.ndarray) -> None:
  n = matrix.shape[0]
  f[f'{name}/nr'] = f[f'{name}/nc'] = np.array([n])
  f[f'{name}/nrg'] = f[f'{name}/ncg'] = np.array([lg.max() + 1])
  f[f'{name}/lc'] = matrix.indices + 1
  f[f'{name}/kr'] = matrix.indptr + 1
  f[f'{name}/lg'] = lg + 1
  f[f'{name}/M'] = matrix.data


def _export(tmp_path, scalar: np.ndarray, nb_jacobian: int | None = None) -> tuple[str, str]:
  nb = scalar.shape[0]
  n = FIELD_BLOCK_COUNT * (nb_jacobian or nb)
  with h5py.File(tmp_path / 'lin_ops.h5', 'w') as f:
    _write_matrix(f, 'jacobian', sparse.csr_array(sparse.eye(n)), np.arange(n))
  with h5py.File(tmp_path / 'stiff_mat.h5', 'w') as f:
    _write_matrix(f, 'stiffmat', sparse.csr_array(scalar), np.arange(nb))
  return str(tmp_path / 'lin_ops.h5'), str(tmp_path / 'stiff_mat.h5')


def _interior_keep(nb: int) -> np.ndarray:
  """Drop vely and by, and each field's two end nodes, as the reduction does."""
  keep = np.zeros(FIELD_BLOCK_COUNT * nb, dtype=bool)
  for block in (0, 1, 3, 4, 5):
    keep[block * nb + 1:(block + 1) * nb - 1] = True
  return keep


def test_reduced_stiffness_is_the_block_laplacian_on_kept_rows(tmp_path) -> None:
  nb = 6
  scalar = _laplacian_1d(nb)
  keep = _interior_keep(nb)

  reduced = reduced_stiffness(*_export(tmp_path, scalar), keep)

  expected = scipy.linalg.block_diag(*([scalar] * FIELD_BLOCK_COUNT))[keep][:, keep]
  np.testing.assert_allclose(reduced.toarray(), expected, rtol=1e-12, atol=1e-14)
  # Interior P1 Laplacian with Dirichlet ends is positive definite.
  assert np.linalg.eigvalsh(reduced.toarray()).min() > 0.0


def test_keeping_a_constrained_row_raises(tmp_path) -> None:
  nb = 6
  keep = _interior_keep(nb)
  keep[0] = True  # density's first node, which the export constrained

  with pytest.raises(ValueError, match='not symmetric'):
    reduced_stiffness(*_export(tmp_path, _laplacian_1d(nb)), keep)


def test_a_mass_matrix_is_not_a_laplacian() -> None:
  check_laplacian(_laplacian_1d(6))
  with pytest.raises(ValueError, match='not a Laplacian'):
    check_laplacian(_mass_1d(6))


def test_a_stiffness_matrix_from_another_mesh_raises(tmp_path) -> None:
  with pytest.raises(ValueError, match='rows'):
    reduced_stiffness(*_export(tmp_path, _laplacian_1d(6), nb_jacobian=7),
                      np.ones(FIELD_BLOCK_COUNT * 7, dtype=bool))


# --- per-mode resolution ------------------------------------------------------


def test_triangular_eigenvectors_satisfy_the_eigenproblem() -> None:
  rng = np.random.default_rng(3)
  t, _ = scipy.linalg.schur(rng.standard_normal((40, 40)).astype(complex), output='complex')
  v = triangular_eigenvectors(t)
  assert np.abs(t @ v - v * np.diag(t)).max() < 1e-12
  assert not np.tril(v, -1).any()


def _diffusion_case(nb: int = 9):
  """L = -D_f M^-1 K per kept field, so each eigenmode lives in one field.

  Its wavenumber there is sqrt(mu) for the matching eigenvalue mu of (K, M),
  exactly, and every other field is absent.
  """
  stiffness = _laplacian_1d(nb)[1:-1, 1:-1]
  mass = _mass_1d(nb)[1:-1, 1:-1]
  kept = (0, 1, 3, 4, 5)
  diffusivity = {0: 1.0, 1: 1.37, 3: 1.91, 4: 2.63, 5: 3.11}
  big_k = scipy.linalg.block_diag(*([stiffness] * len(kept)))
  big_m = scipy.linalg.block_diag(*([mass] * len(kept)))
  generator = -scipy.linalg.block_diag(
    *(diffusivity[b] * np.linalg.solve(mass, stiffness) for b in kept))
  row_fields = np.repeat(np.array(kept), nb - 2)
  scales = {name: 1.0 + i for i, name in enumerate(FIELD_NAMES)}
  norm = Norm(name='test', description='', cache_tag='t',
              factor=scipy.linalg.cholesky(big_m, lower=False), field_scales=scales)
  b = norm.operator(generator)
  t, z = scipy.linalg.schur(b.astype(complex), output='complex')
  mu = scipy.linalg.eigh(stiffness, mass, eigvals_only=True)
  return t, z, norm, sparse.csr_array(big_k), sparse.csr_array(big_m), row_fields, mu, diffusivity


def test_wavenumbers_of_single_field_modes_are_exact() -> None:
  t, z, norm, big_k, big_m, row_fields, mu, diffusivity = _diffusion_case()
  res = mode_resolution(t, z, norm, big_k, big_m, row_fields)

  assert res.fields == (0, 1, 3, 4, 5)
  np.testing.assert_allclose(res.k_max, np.sqrt(mu.max()), rtol=1e-8)
  for i, lam in enumerate(np.diag(t)):
    owner = int(np.argmax(res.share[i]))
    block = res.fields[owner]
    # The mode is -D mu in its own field, and absent from the others.
    expected_mu = -lam.real / diffusivity[block]
    assert res.wavenumber[i, owner] == pytest.approx(np.sqrt(expected_mu), rel=1e-8)
    assert np.isclose(mu, expected_mu, rtol=1e-8).any()
    others = np.delete(res.share[i], owner)
    assert others.max() < ZERO_FIELD_SHARE
  # Absent fields do not vote, so the worst field is the owner.
  owner_ratio = res.wavenumber[np.arange(t.shape[0]), res.share.argmax(axis=1)] / res.k_max[0]
  np.testing.assert_allclose(res.worst(), owner_ratio, rtol=1e-12)


def test_the_worst_field_decides() -> None:
  res = ModeResolution(
    eigenvalues=np.array([-1.0 + 0j, -2.0 + 0j]), fields=(0, 1),
    wavenumber=np.array([[1.0, 8.0], [1.0, 8.0]]),
    share=np.array([[0.999, 1e-3], [1.0, 1e-30]]), k_max=np.array([40.0, 40.0]))
  # A field with 0.1% of the energy still counts; one absent to rounding does not.
  assert res.worst().tolist() == pytest.approx([0.2, 0.025])
  assert res.resolved(0.1).tolist() == [False, True]


def test_the_resolution_cache_is_tied_to_its_schur_factor(tmp_path) -> None:
  nb = 6
  scalar_k = _laplacian_1d(nb)
  n = FIELD_BLOCK_COUNT * nb
  with h5py.File(tmp_path / 'lin_ops.h5', 'w') as f:
    _write_matrix(f, 'jacobian', sparse.csr_array(sparse.eye(n)), np.arange(n))
  with h5py.File(tmp_path / 'stiff_mat.h5', 'w') as f:
    _write_matrix(f, 'stiffmat', sparse.csr_array(scalar_k), np.arange(nb))
  with h5py.File(tmp_path / 'mass_mat.h5', 'w') as f:
    _write_matrix(f, 'massmat', sparse.csr_array(_mass_1d(nb)), np.arange(nb))
  keep = _interior_keep(nb)
  m = int(keep.sum())
  rng = np.random.default_rng(5)
  t, z = scipy.linalg.schur(rng.standard_normal((m, m)).astype(complex), output='complex')
  args = (str(tmp_path / 'lin_ops.h5'), str(tmp_path / 'mass_mat.h5'),
          str(tmp_path / 'stiff_mat.h5'), keep)

  assert load_or_compute_resolution(t, None, L2_COEFFICIENTS, str(tmp_path), *args) is None
  first = load_or_compute_resolution(t, z, L2_COEFFICIENTS, str(tmp_path), *args)
  assert first is not None
  again = load_or_compute_resolution(t, None, L2_COEFFICIENTS, str(tmp_path), *args)
  assert again is not None and np.array_equal(again.wavenumber, first.wavenumber)
  # A different T must not reuse it.
  other = t.copy()
  other[0, 0] += 1e-9
  assert load_or_compute_resolution(other, None, L2_COEFFICIENTS, str(tmp_path), *args) is None
