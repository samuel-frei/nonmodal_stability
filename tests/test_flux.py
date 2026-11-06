"""Sheet-flux observable and flux-optimal seeds, against closed forms and brute force."""

import h5py
import numpy as np
import pytest
import scipy.linalg

from nonmodal.adapters import DofCoordinates, read_dof_coordinates
from nonmodal.fields import FIELD_BLOCK_COUNT, PSI_BLOCK_ID
from nonmodal.flux import (
  adjoint_vector,
  flux_gain_curve,
  flux_optimal_seed,
  flux_response,
  sheet_flux_functional,
  sheet_flux_weights,
)
from nonmodal.norm import L2_COEFFICIENTS, Norm
from nonmodal.subspace import conjugate_closed, restrict_schur

PERIOD = 6.4


def _line_mesh(cells: int, order: int, rows: tuple[float, ...] = (-1.0, 0.0, 1.0),
               nodes: str = 'equispaced') -> DofCoordinates:
  """Nodes of `order`-degree edges along x on each row, periodic, seam folded left."""
  if nodes == 'equispaced':
    local = np.linspace(0.0, 1.0, order + 1)[:-1]
  else:  # Gauss-Lobatto-like, to show nothing assumes equal spacing
    local = 0.5 * (1.0 - np.cos(np.pi * np.arange(order + 1) / order))[:-1]
  h = PERIOD / cells
  x_row = -0.5 * PERIOD + (np.arange(cells)[:, None] * h + local[None, :] * h).ravel()
  x = np.tile(x_row, len(rows))
  z = np.repeat(np.asarray(rows), x_row.size)
  return DofCoordinates(x=x, z=z, x_min=-0.5 * PERIOD, period=PERIOD, plot_file='')


@pytest.mark.parametrize('nodes', ['equispaced', 'lobatto'])
@pytest.mark.parametrize('order', [1, 2, 3])
def test_weights_return_the_cosine_amplitude_on_the_sheet(order: int, nodes: str) -> None:
  mesh = _line_mesh(cells=40, order=order, nodes=nodes)
  w = sheet_flux_weights(mesh, order)
  k = 2.0 * np.pi / PERIOD
  # Only the sheet row carries weight.
  assert np.all(w[mesh.z != 0.0] == 0.0)
  tol = {1: 3e-3, 2: 1e-5, 3: 1e-6}[order]
  assert w @ np.cos(k * mesh.x) == pytest.approx(1.0, abs=tol)
  assert abs(w @ np.sin(k * mesh.x)) < tol
  assert abs(w @ np.cos(2.0 * k * mesh.x)) < tol
  assert abs(w @ np.ones_like(mesh.x)) < 1e-12
  # Shape off the sheet does not matter.
  assert w @ (np.cos(k * mesh.x) * (1.0 + mesh.z**2)) == pytest.approx(1.0, abs=tol)


def test_weights_are_exact_for_the_interpolant() -> None:
  # A field linear on each edge (order 1) is integrated exactly: compare with
  # the trapezoid rule on a much finer subdivision of the same piecewise line.
  mesh = _line_mesh(cells=7, order=1, rows=(0.0,))
  rng = np.random.default_rng(0)
  values = rng.standard_normal(mesh.x.size)
  fine = np.linspace(-0.5 * PERIOD, 0.5 * PERIOD, 70001)
  nodes = np.append(mesh.x, 0.5 * PERIOD)
  interp = np.interp(fine, nodes, np.append(values, values[0]))
  k = 2.0 * np.pi / PERIOD
  reference = 2.0 / PERIOD * np.trapezoid(interp * np.cos(k * fine), fine)
  assert sheet_flux_weights(mesh, 1) @ values == pytest.approx(reference, abs=1e-8)


def test_higher_harmonic_and_offset_line() -> None:
  mesh = _line_mesh(cells=30, order=3, rows=(0.0, 0.5))
  k = 2.0 * np.pi / PERIOD
  w2 = sheet_flux_weights(mesh, 3, harmonic=2, sheet_z=0.5)
  assert np.all(w2[mesh.z != 0.5] == 0.0)
  # Interpolation error scales as (m k h)^(p+1): 16x the m = 1 error here.
  assert w2 @ np.cos(2.0 * k * mesh.x) == pytest.approx(1.0, abs=2e-5)
  assert abs(w2 @ np.cos(k * mesh.x)) < 2e-5


def test_lines_that_are_not_element_edges_are_rejected() -> None:
  mesh = _line_mesh(cells=10, order=3)
  with pytest.raises(ValueError, match='no DOFs lie on'):
    sheet_flux_weights(mesh, 3, sheet_z=0.3)
  with pytest.raises(ValueError, match='not a multiple of order'):
    sheet_flux_weights(mesh, 4)


def _write_plot_and_matrix(tmp_path, points, lg, n_global):  # type: ignore[no-untyped-def]
  plot = tmp_path / 'plot.h5'
  with h5py.File(plot, 'w') as f:
    f.create_dataset('xmhd_2d/smesh/R', data=np.asarray(points, dtype=np.float32))
  mat = tmp_path / 'mass.h5'
  with h5py.File(mat, 'w') as f:
    f.create_dataset('massmat/lg', data=np.asarray(lg) + 1)
    f.create_dataset('massmat/nrg', data=np.array([n_global]))
  return str(plot), str(mat)


def test_dof_coordinates_scatter_through_lg_and_fold_the_seam(tmp_path) -> None:  # type: ignore[no-untyped-def]
  # Four columns of two nodes; local column 4 (x = +2) duplicates global column 0.
  xs, zs = [-2.0, -1.0, 0.0, 1.0, 2.0], [0.0, 1.0]
  points = [(x, z, 0.0) for x in xs for z in zs]
  lg = [(i % 4) * 2 + j for i in range(5) for j in range(2)]
  plot, mat = _write_plot_and_matrix(tmp_path, points, lg, 8)
  coords = read_dof_coordinates(plot, mat)
  assert coords.period == pytest.approx(4.0)
  assert np.allclose(coords.x, np.repeat([-2.0, -1.0, 0.0, 1.0], 2))
  assert np.allclose(coords.z, np.tile([0.0, 1.0], 4))


def test_dof_coordinates_reject_a_plot_that_is_not_the_nodes(tmp_path) -> None:  # type: ignore[no-untyped-def]
  points = [(-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.5, 0.0)]
  plot, mat = _write_plot_and_matrix(tmp_path, points, [0, 1, 0], 2)
  with pytest.raises(ValueError, match='disagree in z'):
    read_dof_coordinates(plot, mat)
  plot, mat = _write_plot_and_matrix(tmp_path, points[:2], [0, 1, 0], 2)
  with pytest.raises(ValueError, match='plot points but'):
    read_dof_coordinates(plot, mat)


# --- the optimal seed -------------------------------------------------------


def _stable(n: int, seed: int, nonnormal: float = 3.0) -> np.ndarray:
  rng = np.random.default_rng(seed)
  a = rng.standard_normal((n, n)) + nonnormal * np.triu(rng.standard_normal((n, n)), 1)
  return a - (np.linalg.eigvals(a).real.max() + 0.5) * np.eye(n)


def _functional(n: int, seed: int):  # type: ignore[no-untyped-def]
  class _F:
    weights = np.random.default_rng(seed).standard_normal(n)
  return _F()


def test_gain_curve_is_the_adjoint_solution() -> None:
  b = _stable(12, 1)
  c = np.random.default_rng(2).standard_normal(12)
  t_mat, z_mat = scipy.linalg.schur(b.astype(complex), output='complex')
  adjoint = z_mat.conj().T @ c
  times = np.array([0.0, 0.1, 0.5, 2.0, 7.0])
  expected = [np.linalg.norm(scipy.linalg.expm(t * b.T) @ c) for t in times]
  assert np.allclose(flux_gain_curve(t_mat, adjoint, times), expected, rtol=1e-10)


def test_seed_attains_the_bound_and_beats_random_initial_conditions() -> None:
  b = _stable(15, 3)
  c = np.random.default_rng(4).standard_normal(15)
  t_mat, z_mat = scipy.linalg.schur(b.astype(complex), output='complex')
  adjoint = z_mat.conj().T @ c
  target = 1.3
  seed = flux_optimal_seed(t_mat, z_mat, adjoint, target, L2_COEFFICIENTS)

  propagator = scipy.linalg.expm(target * b)
  assert np.linalg.norm(seed.vector) == pytest.approx(1.0)
  assert seed.imag_fraction < 1e-12
  assert c @ propagator @ seed.vector == pytest.approx(seed.gain, rel=1e-10)
  assert seed.achieved == pytest.approx(seed.gain, rel=1e-10)
  rng = np.random.default_rng(5)
  for _ in range(200):
    u0 = rng.standard_normal(15)
    assert abs(c @ propagator @ u0) / np.linalg.norm(u0) <= seed.gain * (1 + 1e-12)


def test_restricted_seed_is_real_and_optimal_on_the_subspace() -> None:
  b = _stable(20, 6)
  c = np.random.default_rng(7).standard_normal(20)
  t_full, z_full = scipy.linalg.schur(b.astype(complex), output='complex')
  eig = np.diag(t_full)
  keep = eig.real > np.median(eig.real)  # conjugate pairs share a real part
  assert conjugate_closed(eig[keep])
  t11, z1 = restrict_schur(t_full, z_full, keep)
  adjoint = z1.conj().T @ c
  target = 0.8
  seed = flux_optimal_seed(t11, z1, adjoint, target, L2_COEFFICIENTS)
  assert seed.imag_fraction < 1e-10

  # Brute force over a real orthonormal basis of the invariant subspace.
  q_real, _ = np.linalg.qr(np.hstack([z1.real, z1.imag]))
  q_real = q_real[:, :keep.sum()]
  brute = np.linalg.norm(q_real.T @ scipy.linalg.expm(target * b.T) @ c)
  assert seed.gain == pytest.approx(brute, rel=1e-8)
  assert c @ scipy.linalg.expm(target * b) @ seed.vector == pytest.approx(brute, rel=1e-8)
  # The seed lies in the subspace.
  assert np.linalg.norm(seed.vector - q_real @ (q_real.T @ seed.vector)) < 1e-10


def test_seed_is_measured_in_the_weighted_norm() -> None:
  n = 10
  rng = np.random.default_rng(8)
  generator = _stable(n, 9)
  factor = np.triu(rng.standard_normal((n, n))) + 4.0 * np.eye(n)
  norm = Norm(name='test', description='', cache_tag='t', factor=factor)
  c = rng.standard_normal(n)
  weighted = factor @ generator @ np.linalg.inv(factor)
  t_mat, z_mat = scipy.linalg.schur(weighted.astype(complex), output='complex')

  class _F:
    weights = c
  adjoint = adjoint_vector(_F(), z_mat, norm)  # type: ignore[arg-type]
  target = 0.6
  seed = flux_optimal_seed(t_mat, z_mat, adjoint, target, norm)

  # Brute force: max |c e^{tL} x| / ||R x|| = ||R^-T e^{tL^T} c||.
  expected = np.linalg.norm(np.linalg.solve(factor.T, scipy.linalg.expm(target * generator.T) @ c))
  assert seed.gain == pytest.approx(expected, rel=1e-9)
  assert np.linalg.norm(factor @ seed.vector) == pytest.approx(1.0)
  flux = c @ scipy.linalg.expm(target * generator) @ seed.vector
  assert flux == pytest.approx(expected, rel=1e-9)


def test_response_matches_the_matrix_exponential() -> None:
  b = _stable(9, 10)
  c = np.random.default_rng(11).standard_normal(9)
  t_mat, z_mat = scipy.linalg.schur(b.astype(complex), output='complex')
  adjoint = z_mat.conj().T @ c
  seed = flux_optimal_seed(t_mat, z_mat, adjoint, 0.5, L2_COEFFICIENTS)
  times = np.array([0.0, 0.25, 0.5, 3.0])
  flux, size = flux_response(t_mat, adjoint, seed.schur_vector, times)
  for t, f, s in zip(times, flux, size, strict=True):
    u = scipy.linalg.expm(t * b) @ seed.vector
    assert f == pytest.approx(c @ u, rel=1e-10, abs=1e-14)
    assert s == pytest.approx(np.linalg.norm(u), rel=1e-10)
  assert flux[2] == pytest.approx(seed.gain, rel=1e-10)


def test_functional_lives_in_the_psi_block() -> None:
  mesh = _line_mesh(cells=8, order=2)
  n = mesh.x.size
  keep = np.ones(FIELD_BLOCK_COUNT * n, dtype=bool)
  keep[2 * n:3 * n] = False  # drop a whole block, as the reduction drops vely
  functional = sheet_flux_functional(mesh, 2, keep, scale=2.0)
  full = np.zeros(keep.size)
  full[keep] = functional.weights
  psi = full[PSI_BLOCK_ID * n:(PSI_BLOCK_ID + 1) * n]
  assert np.allclose(psi, sheet_flux_weights(mesh, 2) / 2.0)
  assert np.count_nonzero(full) == functional.n_nodes == np.count_nonzero(psi)
  assert functional.describe()['wavenumber_per_m'] == pytest.approx(2.0 * np.pi / PERIOD)

  keep[PSI_BLOCK_ID * n + int(np.flatnonzero(mesh.z == 0.0)[0])] = False
  with pytest.raises(ValueError, match='not in the reduced state'):
    sheet_flux_functional(mesh, 2, keep, scale=2.0)
