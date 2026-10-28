"""The continuous-time generator recovered from a backward-Euler Jacobian."""

import numpy as np
import pytest
from scipy import sparse

from nonmodal.operator import (
  GENERATOR_CACHE,
  LEGACY_CACHES,
  SCHUR_CACHE,
  SCHURVEC_CACHE,
  check_timestep,
  generator_from_matrices,
)


def _mass_and_stiffness(n: int = 12, seed: int = 3):
  """An SPD mass matrix and a non-symmetric K, shaped like an FE discretisation."""
  rng = np.random.default_rng(seed)
  mass = sparse.diags([1.0, 4.0, 1.0], [-1, 0, 1], shape=(n, n)).toarray() / 6.0
  lap = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)).toarray()
  advect = sparse.diags([-1.0, 1.0], [-1, 1], shape=(n, n)).toarray()
  stiffness = 50.0 * lap + 30.0 * advect + rng.standard_normal((n, n))
  return sparse.csr_array(mass), stiffness


@pytest.mark.parametrize('dt', [1e-7, 1e-3, 0.5])
def test_the_generator_is_minus_m_inverse_k(dt: float) -> None:
  """J = M + dt*K, so (M - J)/dt solved against M is exactly -M^-1 K."""
  mass, stiffness = _mass_and_stiffness()
  jac = sparse.csr_array(mass.toarray() + dt * stiffness)
  expected = -np.linalg.solve(mass.toarray(), stiffness)

  got = generator_from_matrices(jac, mass, dt)

  np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-8 * np.abs(expected).max())


def test_the_generator_does_not_depend_on_the_export_timestep() -> None:
  mass, stiffness = _mass_and_stiffness()
  generators = [
    generator_from_matrices(sparse.csr_array(mass.toarray() + dt * stiffness), mass, dt)
    for dt in (1e-7, 2e-7, 5e-7)]
  for other in generators[1:]:
    np.testing.assert_allclose(other, generators[0], rtol=1e-6)


def test_the_old_operator_is_the_backward_euler_image_of_the_generator() -> None:
  """(J^-1 M - I)/dt has eigenvalues lambda/(1 - dt*lambda) of L's: what L replaces."""
  mass, stiffness = _mass_and_stiffness()
  dt = 1e-2
  jac = mass.toarray() + dt * stiffness
  old = (np.linalg.solve(jac, mass.toarray()) - np.eye(jac.shape[0])) / dt
  lam = np.linalg.eigvals(generator_from_matrices(sparse.csr_array(jac), mass, dt))

  mapped = lam / (1.0 - dt * lam)
  old_eigs = np.linalg.eigvals(old)
  assert np.abs(old_eigs[:, None] - mapped[None, :]).min(axis=1).max() < 1e-8 * np.abs(mapped).max()
  # And the map is far from the identity for the stiff end of this spectrum.
  assert np.abs(mapped - lam).max() > 0.1 * np.abs(lam).max()


def test_a_timestep_that_differs_from_the_deck_is_rejected() -> None:
  check_timestep(1e-7, 1.0e-7, 'oft.in')
  with pytest.raises(ValueError, match='differs from dt=1e-07'):
    check_timestep(2e-7, 1e-7, 'oft.in')


def test_generator_caches_never_share_a_name_with_old_operator_caches() -> None:
  """Caches are keyed by filename alone, so a shared name would reuse A as L."""
  assert not {GENERATOR_CACHE, SCHUR_CACHE, SCHURVEC_CACHE} & set(LEGACY_CACHES)
